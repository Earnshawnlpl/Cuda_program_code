#include <iostream>
#include <string>
#include <random>   // 用于随机数
#include <cstring>  // 用于 memset
#include <chrono>   
#include <iomanip>
#include <cublas_v2.h>
#include <math.h>

// 辅助宏：检查 CUDA 错误
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void verify_results(const float* h_inp, const float* h_out, int N, int C, const char* label) {
    printf("\n--- [%s] 验证结果 (N=%d, C=%d) ---\n", label, N, C);

    for (int i = 0; i < N; i++) {
        printf("样本 %d:\n", i);
        
        double row_sum = 0.0;
        
        // 打印输入 (Logits)
        printf("  输入 (Logits): ");
        for (int j = 0; j < C; j++) {
            printf("%8.2f ", h_inp[i * C + j]);
        }

        // 打印输出 (Probs) 并计算概率和
        printf("\n  输出 (Probs):  ");
        for (int j = 0; j < C; j++) {
            float val = h_out[i * C + j];
            printf("%8.4f ", val);
            row_sum += (double)val;
        }
        
        // 验证概率之和是否为 1
        printf(" | 概率和: %.4f %s\n", 
               row_sum, 
               (fabs(row_sum - 1.0) < 1e-5 ? "\033[32m(通过)\033[0m" : "\033[31m(失败)\033[0m"));
    }
}

