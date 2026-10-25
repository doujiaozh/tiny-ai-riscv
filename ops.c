#include "ops.h"
#include <stdio.h>

// 点积：计算两个向量的点积
void dot_product(const tensor_t* a, const tensor_t* b, tensor_t* result) {
    if (a->dims != 1 || b->dims != 1 || a->shape[0] != b->shape[0] || result->dims != 1) {
        return; // 简化处理：只支持一维向量
    }
    
    float sum = 0.0f;
    for (int i = 0; i < a->shape[0]; i++) {
        sum += a->data[i] * b->data[i];
    }
    result->data[0] = sum;
}

// 矩阵乘法：a(m×n) × b(n×p) = result(m×p)
void matmul(const tensor_t* a, const tensor_t* b, tensor_t* result) {
    if (a->dims != 2 || b->dims != 2 || result->dims != 2) {
        return; // 简化处理：只支持二维矩阵
    }
    if (a->shape[1] != b->shape[0]) {
        return; // 维度不匹配
    }
    
    int m = a->shape[0];
    int n = a->shape[1];
    int p = b->shape[1];
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }
            result->data[i * p + j] = sum;
        }
    }
}

// ReLU 激活函数：对张量每个元素应用 max(0, x)
void relu(const tensor_t* input, tensor_t* output) {
    if (input->dims != output->dims) return;
    
    size_t size = 1;
    for (int i = 0; i < input->dims; i++) {
        size *= input->shape[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0.0f;
    }
}

// 加法：对两个相同形状的张量相加
void add(const tensor_t* a, const tensor_t* b, tensor_t* result) {
    size_t size = 1;
    for (int i = 0; i < a->dims; i++) {
        size *= a->shape[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

// 缩放：将张量乘以常数
void scale(tensor_t* t, float factor) {
    size_t size = 1;
    for (int i = 0; i < t->dims; i++) {
        size *= t->shape[i];
    }
    
    for (size_t i = 0; i < size; i++) {
        t->data[i] *= factor;
    }
}
