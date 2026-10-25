#ifndef NN_H
#define NN_H

#include "tensor.h"
#include "ops.h"

// 定义一个简单的 2 层神经网络
// 结构：Input(4) -> Hidden(8) -> Output(2)
// 使用 tanh 和 ReLU 作为激活函数

typedef struct {
    // 权重矩阵
    tensor_t* W1;  // 4x8 权重矩阵
    tensor_t* W2;  // 8x2 权重矩阵
    
    // 偏置向量
    tensor_t* b1;  // 8 维偏置
    tensor_t* b2;  // 2 维偏置
    
} neural_network_t;

// 初始化网络（硬编码权重，实际应用中应加载训练好的权重）
void nn_init(neural_network_t* nn);

// 前向传播：input -> output
void nn_forward(const neural_network_t* nn, const tensor_t* input, tensor_t* output);

// 销毁网络
void nn_destroy(neural_network_t* nn);

#endif // NN_H
