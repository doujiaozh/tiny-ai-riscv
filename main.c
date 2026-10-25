/* ============================================================
 * 纯 C 微型推理引擎 —— 测试主程序
 * 零依赖、可交叉编译到 RISC-V 裸机
 * ============================================================ */
#include <stdio.h>
#include <string.h>

#include "tensor.h"
#include "ops.h"
#include "nn.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do {                              \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); }  \
    else      { g_fail++; printf("  [FAIL] %s\n", msg); }  \
} while (0)

static float fabsf_local(float x) { return x < 0.0f ? -x : x; }
static int   nearly_equal(float a, float b, float eps) {
    return fabsf_local(a - b) < eps;
}

/* ---------- 测试 1：张量创建与多维索引 ---------- */
static void test_tensor(void) {
    printf("\n[测试 1] 张量数据结构（多维数组）\n");

    int shape[2] = {3, 4};
    tensor_t* m = tensor_create(2, shape);
    CHECK(m != NULL, "创建 3x4 张量");
    CHECK(m->dims == 2 && m->shape[0] == 3 && m->shape[1] == 4, "维度/形状正确");

    /* 全零初始化 */
    int all_zero = 1;
    for (int i = 0; i < 12; i++) {
        if (m->data[i] != 0.0f) { all_zero = 0; break; }
    }
    CHECK(all_zero, "零初始化");

    /* 多维索引读写：(1,2) 在行主序下是第 1*4+2 = 6 个元素 */
    int idx_a[2] = {1, 2};
    tensor_set(m, idx_a, 6.5f);
    CHECK(nearly_equal(tensor_get(m, idx_a), 6.5f, 1e-6f), "tensor_set/tensor_get (1,2)");
    CHECK(nearly_equal(m->data[6], 6.5f, 1e-6f), "行主序线性索引 data[6] == (1,2)");

    int idx_b[2] = {2, 0};
    tensor_set(m, idx_b, 9.0f);
    CHECK(nearly_equal(m->data[8], 9.0f, 1e-6f), "行主序线性索引 data[8] == (2,0)");

    tensor_print(m, "  3x4 矩阵");
    tensor_destroy(m);
}

/* ---------- 测试 2：点积 ---------- */
static void test_dot_product(void) {
    printf("\n[测试 2] 点积 Dot Product\n");

    int v[1] = {3};
    tensor_t* a = tensor_create(1, v);
    tensor_t* b = tensor_create(1, v);
    tensor_t* r = tensor_create(1, (int[1]){1});

    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;
    b->data[0] = 4.0f; b->data[1] = 5.0f; b->data[2] = 6.0f;

    dot_product(a, b, r);
    /* 1*4 + 2*5 + 3*6 = 32 */
    CHECK(nearly_equal(r->data[0], 32.0f, 1e-5f), "dot([1,2,3],[4,5,6]) == 32");
    printf("  结果 = %.4f (期望 32.0000)\n", r->data[0]);

    tensor_destroy(a);
    tensor_destroy(b);
    tensor_destroy(r);
}

/* ---------- 测试 3：矩阵乘法 ---------- */
static void test_matmul(void) {
    printf("\n[测试 3] 矩阵乘法 MatMul\n");

    tensor_t* A = tensor_create(2, (int[2]){2, 3});
    tensor_t* B = tensor_create(2, (int[2]){3, 2});
    tensor_t* C = tensor_create(2, (int[2]){2, 2});

    /* A = [[1,2,3],[4,5,6]] */
    A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
    A->data[3] = 4.0f; A->data[4] = 5.0f; A->data[5] = 6.0f;
    /* B = [[7,8],[9,10],[11,12]] */
    B->data[0] = 7.0f;  B->data[1] = 8.0f;
    B->data[2] = 9.0f;  B->data[3] = 10.0f;
    B->data[4] = 11.0f; B->data[5] = 12.0f;

    matmul(A, B, C);
    /* C = [[58,64],[139,154]] */
    CHECK(nearly_equal(C->data[0], 58.0f, 1e-4f), "C[0][0] == 58");
    CHECK(nearly_equal(C->data[1], 64.0f, 1e-4f), "C[0][1] == 64");
    CHECK(nearly_equal(C->data[2], 139.0f, 1e-4f), "C[1][0] == 139");
    CHECK(nearly_equal(C->data[3], 154.0f, 1e-4f), "C[1][1] == 154");

    printf("  A(2x3) x B(3x2) =\n");
    printf("    [[%.0f, %.0f],\n", C->data[0], C->data[1]);
    printf("     [%.0f, %.0f]]  (期望 [[58,64],[139,154]])\n", C->data[2], C->data[3]);

    tensor_destroy(A);
    tensor_destroy(B);
    tensor_destroy(C);
}

/* ---------- 测试 4：ReLU ---------- */
static void test_relu(void) {
    printf("\n[测试 4] ReLU 激活函数\n");

    tensor_t* x = tensor_create(1, (int[1]){5});
    tensor_t* y = tensor_create(1, (int[1]){5});

    x->data[0] = -2.0f; x->data[1] = -1.0f; x->data[2] = 0.0f;
    x->data[3] =  1.0f; x->data[4] =  2.0f;

    relu(x, y);

    CHECK(nearly_equal(y->data[0], 0.0f, 1e-6f), "ReLU(-2) == 0");
    CHECK(nearly_equal(y->data[1], 0.0f, 1e-6f), "ReLU(-1) == 0");
    CHECK(nearly_equal(y->data[2], 0.0f, 1e-6f), "ReLU( 0) == 0");
    CHECK(nearly_equal(y->data[3], 1.0f, 1e-6f), "ReLU( 1) == 1");
    CHECK(nearly_equal(y->data[4], 2.0f, 1e-6f), "ReLU( 2) == 2");

    tensor_print(x, "  输入");
    tensor_print(y, "  ReLU 输出");

    tensor_destroy(x);
    tensor_destroy(y);
}

/* ---------- 测试 5：2 层神经网络前向传播 ---------- */
static void test_neural_network(void) {
    printf("\n[测试 5] 2 层神经网络前向传播\n");
    printf("  结构: Input(4) -> Linear -> ReLU -> Hidden(8) -> Linear -> Output(2)\n");

    neural_network_t nn;
    nn_init(&nn);

    tensor_t* input  = tensor_create(2, (int[2]){1, 4});
    tensor_t* output = tensor_create(2, (int[2]){1, 2});

    /* 4 组测试输入 */
    static const float X[4][4] = {
        { 0.5f, -0.3f,  0.8f, -0.1f},
        { 1.0f,  0.0f,  1.0f,  0.0f},
        { 0.5f,  0.5f, -0.5f, -0.5f},
        {-0.8f,  0.3f, -0.2f,  0.9f}
    };

    /* 不同输入必须产生不同输出（验证前向传播确实在传播信息）*/
    float outs[4][2];
    int distinct = 0;
    for (int t = 0; t < 4; t++) {
        memcpy(input->data, X[t], sizeof(float) * 4);
        nn_forward(&nn, input, output);
        outs[t][0] = output->data[0];
        outs[t][1] = output->data[1];
        if (t > 0) {
            if (!nearly_equal(outs[t][0], outs[t-1][0], 1e-4f) ||
                !nearly_equal(outs[t][1], outs[t-1][1], 1e-4f)) {
                distinct++;
            }
        }
    }
    CHECK(distinct >= 2, "不同输入产生不同输出（前向传播有效）");

    /* 打印输入/输出对比表 */
    printf("\n  ┌────────────────────────────────────┬───────────────────────────┐\n");
    printf("  │              输入 (x1,x2,x3,x4)    │      输出 (y1,y2)         │\n");
    printf("  ├────────────────────────────────────┼───────────────────────────┤\n");
    for (int t = 0; t < 4; t++) {
        printf("  │ [%6.2f %6.2f %6.2f %6.2f ] │  [ %9.4f, %9.4f ] │\n",
               X[t][0], X[t][1], X[t][2], X[t][3],
               outs[t][0], outs[t][1]);
    }
    printf("  └────────────────────────────────────┴───────────────────────────┘\n");

    /* 打印第一组输入的逐层中间值，便于人工核对 */
    printf("\n  逐层分解（第 1 组输入）:\n");
    memcpy(input->data, X[0], sizeof(float) * 4);
    tensor_t* z1 = tensor_create(2, (int[2]){1, 8});
    tensor_t* h  = tensor_create(2, (int[2]){1, 8});
    matmul(input, nn.W1, z1);
    add(z1, nn.b1, z1);
    tensor_print(z1, "    z1 = x@W1+b1");
    relu(z1, h);
    tensor_print(h, "    h  = ReLU(z1)");
    tensor_destroy(z1);
    tensor_destroy(h);

    nn_destroy(&nn);
    tensor_destroy(input);
    tensor_destroy(output);
}

/* ---------- 测试 6：随机初始化张量（验证可复现性）---------- */
static void test_random_tensor(void) {
    printf("\n[测试 6] 随机初始化张量（确定性伪随机，可复现）\n");

    tensor_t* a = tensor_random(2, (int[2]){2, 4}, 0.7f);
    tensor_t* b = tensor_random(2, (int[2]){2, 4}, 0.7f);

    CHECK(a != NULL && b != NULL, "创建随机张量");

    int same = 1;
    for (int i = 0; i < 8; i++) {
        if (!nearly_equal(a->data[i], b->data[i], 1e-6f)) { same = 0; break; }
    }
    CHECK(same, "相同种子产生相同序列（确定性初始化）");

    tensor_print(a, "  随机张量 a");

    tensor_destroy(a);
    tensor_destroy(b);
}

int main(void) {
    printf("============================================================\n");
    printf("   纯 C 微型推理引擎  v1.0   (零依赖 / RISC-V 友好)\n");
    printf("============================================================\n");

    test_tensor();
    test_dot_product();
    test_matmul();
    test_relu();
    test_neural_network();
    test_random_tensor();

    printf("\n============================================================\n");
    printf("  测试结果: %d 通过, %d 失败\n", g_pass, g_fail);
    printf("============================================================\n");

    return g_fail == 0 ? 0 : 1;
}
