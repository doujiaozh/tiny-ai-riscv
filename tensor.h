#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// 定义最大维度数
#define MAX_DIMS 8

// 定义浮点类型，适合嵌入式
typedef float tensor_float;

// 张量结构
typedef struct {
    int dims;                        // 维度数量
    int shape[MAX_DIMS];             // 每个维度的大小
    tensor_float* data;              // 数据指针（一维存储）
} tensor_t;

// 创建零初始化张量
tensor_t* tensor_create(int dims, const int* shape);

// 创建随机初始化张量
tensor_t* tensor_random(int dims, const int* shape, float scale);

// 销毁张量
void tensor_destroy(tensor_t* t);

// 获取单个元素
tensor_float tensor_get(const tensor_t* t, int* indices);

// 设置单个元素
void tensor_set(tensor_t* t, int* indices, tensor_float value);

// 打印张量
void tensor_print(const tensor_t* t, const char* name);

#endif // TENSOR_H
