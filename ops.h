#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// 点积：计算两个向量的点积
void dot_product(const tensor_t* a, const tensor_t* b, tensor_t* result);

// 矩阵乘法：a(m×n) × b(n×p) = result(m×p)
void matmul(const tensor_t* a, const tensor_t* b, tensor_t* result);

// ReLU 激活函数：对张量每个元素应用 max(0, x)
void relu(const tensor_t* input, tensor_t* output);

// 加法：对两个相同形状的张量相加
void add(const tensor_t* a, const tensor_t* b, tensor_t* result);

// 缩放：将张量乘以常数
void scale(tensor_t* t, float factor);

#endif // OPS_H
