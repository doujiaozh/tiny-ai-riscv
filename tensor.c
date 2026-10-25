#include "tensor.h"
#include <stdio.h>

// 辅助函数：计算单个元素的线性索引
static int calc_index(const tensor_t* t, int* indices) {
    int index = 0;
    int multiplier = 1;
    
    // 从最内层维度开始计算
    for (int i = t->dims - 1; i >= 0; i--) {
        index += indices[i] * multiplier;
        multiplier *= t->shape[i];
    }
    
    return index;
}

// 创建零初始化张量
tensor_t* tensor_create(int dims, const int* shape) {
    if (dims > MAX_DIMS || dims < 1) {
        return NULL;
    }
    
    tensor_t* t = (tensor_t*)malloc(sizeof(tensor_t));
    if (!t) return NULL;
    
    t->dims = dims;
    memcpy(t->shape, shape, dims * sizeof(int));
    
    // 计算总元素数
    size_t size = 1;
    for (int i = 0; i < dims; i++) {
        size *= shape[i];
    }
    
    t->data = (tensor_float*)calloc(size, sizeof(tensor_float));
    if (!t->data) {
        free(t);
        return NULL;
    }
    
    return t;
}

// 创建随机初始化张量
tensor_t* tensor_random(int dims, const int* shape, float scale) {
    tensor_t* t = tensor_create(dims, shape);
    if (!t) return NULL;
    
    size_t size = 1;
    for (int i = 0; i < dims; i++) {
        size *= shape[i];
    }
    
    // 简单的伪随机数生成
    unsigned int seed = 12345;
    for (size_t i = 0; i < size; i++) {
        seed = seed * 1103515245 + 12345;
        t->data[i] = ((seed >> 16) & 0x7FFF) / 32767.0f * scale * 2 - scale;
    }
    
    return t;
}

// 销毁张量
void tensor_destroy(tensor_t* t) {
    if (t) {
        free(t->data);
        free(t);
    }
}

// 获取单个元素（非 static，可链接）
tensor_float tensor_get(const tensor_t* t, int* indices) {
    int index = calc_index(t, indices);
    return t->data[index];
}

// 设置单个元素（非 static，可链接）
void tensor_set(tensor_t* t, int* indices, tensor_float value) {
    int index = calc_index(t, indices);
    t->data[index] = value;
}

// 打印张量
void tensor_print(const tensor_t* t, const char* name) {
    // 只显示前 20 个元素
    size_t print_count = 20;
    size_t total = 1;
    for (int i = 0; i < t->dims; i++) {
        total *= t->shape[i];
    }
    if (print_count > total) print_count = total;
    
    if (name && name[0] != '\0') {
        printf("%s: ", name);
    }
    printf("[");
    for (size_t i = 0; i < print_count; i++) {
        printf(" %.4f", t->data[i]);
    }
    if (total > print_count) {
        printf(" ...");
    }
    printf("] [%d elements]\n", (int)total);
}
