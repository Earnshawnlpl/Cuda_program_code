#include <iostream>
#include <string>
#include <random>   // 用于随机数
#include <cstring>  // 用于 memset
#include <chrono>   
#include <iomanip>
#include <cublas_v2.h>
#include <math.h>
#include "common.h"

/**
 * CPU 版 Softmax 前向传播函数
 * @param out 指向输出缓冲区的指针，大小为 (N * C)
 * @param inp 指向输入数据的指针（Logits），大小为 (N * C)
 * @param N   Batch Size（批大小），即有多少行数据
 * @param C   Channels（通道数），即每一行有多少个类别
 */
void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    // 外部循环：遍历 Batch 中的每一行（每一个样本）
    for (int i = 0; i < N; i++) {
        // 计算当前行的起始位置指针
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        // --- 第一步：寻找当前行的最大值 (Max Trick) ---
        // 目的：防止后续 expf(x) 计算时出现数值溢出 (Overflow)
        float maxval = -INFINITY; 
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }

        // --- 第二步：计算指数和 (Sum of Exponentials) ---
        // 注意：这里使用了 double 类型来累加 sum。
        // 理由：为了确保 CPU 计算出的结果足够精确，可以作为 CUDA 核函数的“地面真值 (Ground-truth)”进行对比验证。
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            // 减去 maxval 后再取指数，保证指数项最大为 e^0 = 1
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        // --- 第三步：归一化 (Normalization) ---
        // 先计算总和的倒数，将除法转为乘法以提高计算效率
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            // 最终每个类别的概率：e^(x-max) / sum(e^(x-max))
            out_row[j] *= norm;
        }
    }
}

void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        float sum = 0.0f;
		for (int j = 0; j < C; j++) {
			float maxval_prev = maxval;
			if (inp_row[j] > maxval) {
				maxval = inp_row[j];
				sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
			} else {
				sum += expf(inp_row[j] - maxval);
			}
		}

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }
        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

bool check_results(float* out1, float* out2, int N, int C, float threshold = 1e-6f) {
    float max_diff = 0.0f;
    for (int i = 0; i < N * C; i++) {
        float diff = fabsf(out1[i] - out2[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("最大绝对误差: %e\n", max_diff);
    return max_diff < threshold;
}

int main() {
    // 1. 设置维度参数
    int N = 4096; // 批大小 (行)
    int C = 512;  // 通道数 (列)

    // 2. 动态分配内存 (C 语言使用 malloc)
    float* h_inp = (float*)malloc(N * C * sizeof(float));
    float* h_out_naive = (float*)malloc(N * C * sizeof(float));
    float* h_out_cpu_online = (float*)malloc(N * C * sizeof(float));
    float* h_out_kernel1 = (float*)malloc(N * C * sizeof(float));

    // 3. 初始化输入数据 (使用随机数)
    srand((unsigned int)time(NULL));
    for (int i = 0; i < N * C; i++) {
        // 生成 -2.0 到 2.0 之间的随机数
        h_inp[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
    }

    // 为了测试数值稳定性，手动设置一个较大的值
    h_inp[0] = 100.0f;

    printf("--- Softmax 计算开始 (N=%d, C=%d) ---\n", N, C);


    /*          naive——softmax                */
    // 4. 调用 Softmax 函数
    auto start = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(h_out_naive, h_inp, N, C);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_naive = end - start;

    /*          online softmax                */
    start = std::chrono::high_resolution_clock::now();
    softmax_forward_online_cpu(h_out_cpu_online, h_inp, N, C);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_online = end - start;


    size_t size = N * C * sizeof(float);
    float *d_inp, *d_out;
    //内存分配
    CHECK_CUDA(cudaMalloc(&d_inp, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    CHECK_CUDA(cudaMemcpy(d_inp, h_inp, size, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    start = std::chrono::high_resolution_clock::now();
    softmax_forward_kernel1<<<grid, block>>>(d_out, d_inp, N, C);
    CHECK_CUDA(cudaDeviceSynchronize()); // 必须同步才能准确计时
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_kernel1 = end - start;

    // 拷贝结果
    CHECK_CUDA(cudaMemcpy(h_out_kernel1, d_out, size, cudaMemcpyDeviceToHost));

    // --- 输出结果 ---
    printf("Naive  Softmax 耗时: %f ms\n", ms_naive.count());
    printf("Online Softmax 耗时: %f ms\n", ms_online.count());
    printf("kernel1 Softmax 耗时: %f ms\n", ms_kernel1.count());

    if (check_results(h_out_naive, h_out_kernel1, N, C)) {
        printf("结果验证通过：两者一致。\n");
    } else {
        printf("结果验证失败：误差过大！\n");
    }


    // 6. 释放内存
    free(h_inp);
    free(h_out_naive);
    free(h_out_cpu_online);
    free(h_out_kernel1);
    CHECK_CUDA(cudaFree(d_inp));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}