#include "nn.h"
#include <stdio.h>
#include <string.h>

/* ============================================================
 * 硬编码的"随机初始化"权重
 * 说明：裸机环境没有 rand()/srand()，这里把一组确定性伪
 * 随机数（种子 12345，LGC 生成，范围 [-0.7, 0.7]）直接
 * 写进代码，既满足"硬编码随机权重"的要求，又保证在任意
 * 平台上结果完全一致、可复现。
 * ============================================================ */

/* W1: 输入层(4) -> 隐藏层(8) 权重，形状 4x8，行主序 */
static const float W1_DATA[4 * 8] = {
    0.2294f, -0.2171f,  0.5106f, -0.1195f,  0.6156f, -0.3346f,  0.2629f, -0.2976f,
   -0.5932f,  0.3876f, -0.2326f, -0.4138f, -0.0065f, -0.2157f, -0.1249f,  0.5014f,
    0.2061f,  0.2732f, -0.1897f,  0.5627f,  0.0475f,  0.1626f, -0.1386f,  0.0281f,
   -0.0682f,  0.1427f,  0.1103f, -0.0187f, -0.1241f, -0.5331f, -0.0724f, -0.3406f
};

/* W2: 隐藏层(8) -> 输出层(2) 权重，形状 8x2，行主序 */
static const float W2_DATA[8 * 2] = {
    0.4114f, -0.3065f,
    0.0251f,  0.3691f,
   -0.5129f, -0.2020f,
   -0.2861f, -0.1508f,
    0.3815f, -0.3155f,
   -0.1962f,  0.0997f,
    0.0892f, -0.3188f,
   -0.0426f,  0.4992f
};

/* b1: 隐藏层偏置，形状 1x8 */
static const float B1_DATA[1 * 8] = {
    0.2133f,  0.2424f, -0.2456f, -0.2670f,
    0.3638f, -0.0374f,  0.0366f,  0.3581f
};

/* b2: 输出层偏置，形状 1x2 */
static const float B2_DATA[1 * 2] = {
    0.1048f, -0.0962f
};

/* 初始化网络：把硬编码权重装进张量 */
void nn_init(neural_network_t* nn) {
    int w1_shape[2] = {4, 8};
    int w2_shape[2] = {8, 2};
    int b1_shape[2] = {1, 8};
    int b2_shape[2] = {1, 2};

    nn->W1 = tensor_create(2, w1_shape);
    nn->W2 = tensor_create(2, w2_shape);
    nn->b1 = tensor_create(2, b1_shape);
    nn->b2 = tensor_create(2, b2_shape);

    if (!nn->W1 || !nn->W2 || !nn->b1 || !nn->b2) {
        printf("[nn_init] 内存分配失败\n");
        return;
    }

    memcpy(nn->W1->data, W1_DATA, sizeof(W1_DATA));
    memcpy(nn->W2->data, W2_DATA, sizeof(W2_DATA));
    memcpy(nn->b1->data, B1_DATA, sizeof(B1_DATA));
    memcpy(nn->b2->data, B2_DATA, sizeof(B2_DATA));
}

/* 前向传播（batch = 1）：
 *   z1 = x @ W1 + b1      (1x4 @ 4x8 + 1x8 -> 1x8)
 *   h  = ReLU(z1)         (1x8)
 *   y  = h  @ W2 + b2     (1x8 @ 8x2 + 1x2 -> 1x2)   线性输出
 *
 * 输入  input : 形状 1x4
 * 输出  output: 形状 1x2
 */
void nn_forward(const neural_network_t* nn, const tensor_t* input, tensor_t* output) {
    int hidden_shape[2] = {1, 8};

    tensor_t* z1 = tensor_create(2, hidden_shape);  /* 第一层线性结果 */
    tensor_t* h  = tensor_create(2, hidden_shape);  /* ReLU 激活结果  */

    if (!z1 || !h) {
        printf("[nn_forward] 内存分配失败\n");
        if (z1) tensor_destroy(z1);
        if (h)  tensor_destroy(h);
        return;
    }

    /* ---- 第 1 层：Linear + ReLU ---- */
    matmul(input, nn->W1, z1);   /* z1 = x @ W1     */
    add(z1, nn->b1, z1);         /* z1 = z1 + b1    */
    relu(z1, h);                 /* h  = ReLU(z1)   */

    /* ---- 第 2 层：Linear（输出层不加激活，直接输出 logits）---- */
    matmul(h, nn->W2, output);   /* y = h @ W2      */
    add(output, nn->b2, output); /* y = y + b2      */

    tensor_destroy(z1);
    tensor_destroy(h);
}

/* 销毁网络 */
void nn_destroy(neural_network_t* nn) {
    tensor_destroy(nn->W1);
    tensor_destroy(nn->W2);
    tensor_destroy(nn->b1);
    tensor_destroy(nn->b2);
    nn->W1 = nn->W2 = nn->b1 = nn->b2 = NULL;
}
