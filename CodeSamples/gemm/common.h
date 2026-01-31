#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 1. 处理 CUDA Runtime API 的版本 (如 cudaMalloc)
static void check(cudaError_t result, char const* const func, const char* const file, const int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d\n", file, line);
        fprintf(stderr, "Code: %d, Name: %s, Function: %s\n", 
                (int)result, cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

// 2. 处理 cuBLAS API 的版本 (如 cublasSgemm)
static void check(cublasStatus_t result, char const* const func, const char* const file, const int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error at %s:%d\n", file, line);
        fprintf(stderr, "Code: %d, Function: %s\n", (int)result, func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)