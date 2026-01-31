#include <iostream>
#include <string>
#include <random>   // 用于随机数
#include <cstring>  // 用于 memset
#include "common.h"
#include <chrono>   
#include <iomanip>
#include <cublas_v2.h>
#include <math.h>
#define CEIL(A, B) (((A) + (B) - 1) / (B))


__global__ void naive_gemm(float *A, float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N)
    {
        float tmp = 0.0;
        for(int i = 0; i < K; i++)
        {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = tmp;
    }
}


void serial_gemm(float* A,float* B,float* C, int M,int N, int K)
{
    for(int m = 0; m < M;m++)
    {
        for(int n = 0; n < N;n++)
        {
            for(int k = 0; k < K;k++)
            {
                C[m * N + n] += A[m * K + k] * B[k * N + n]; 
            }
        }
    }
}

// 每个 Block 负责计算 C 中 (Bm, Bn) 大小的分块 tileC
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
         int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void blockTileGEMM(float* A, float* B, float* C, const int M, const int K, const int N) {
  __shared__ float As[Bm][Bk];  // 存储 tileA
  __shared__ float Bs[Bk][Bn];  // 存储 tileB

  // 计算 block 负责的 tileC 左上角元素的行列坐标
  int r0 = blockIdx.y * Bm;
  int c0 = blockIdx.x * Bn;

  // 当前 thread 的编号（默认为一维 block 配置）
  int tid = threadIdx.x;

  /*------ tileA ------*/
  // 写入 A tile 时，block 中 thread 排布尺寸为 (A_BLOCK_X, blockSize / A_BLOCK_X) = (8, 32)
  constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

  // 对于 tid 号线程，其位于 blockA 中的行列坐标为 (tid / A_BLOCK_X, tid % A_BLOCK_X)
  int A_THREAD_Y = tid / A_BLOCK_X;
  int A_THREAD_X = tid % A_BLOCK_X;

  /*------ tileB ------*/
  // 写入 B tile 时，block 中 thread 排布尺寸为 (B_BLOCK_X, blockSize / B_BLOCK_X) = (32, 8)
  constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;

  // 对于 tid 号线程，其位于 blockB 中的行列坐标为 (tid / B_BLOCK_X, tid % B_BLOCK_X)
  int B_THREAD_Y = tid / B_BLOCK_X;
  int B_THREAD_X = tid % B_BLOCK_X;

  /*------ tileC ------*/
  constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;

  // 对于 tid 号线程，其位于 blockC 中的行列坐标为 (tid / C_BLOCK_X, tid % C_BLOCK_X)
  int C_THREAD_Y = tid / C_BLOCK_X;
  int C_THREAD_X = tid % C_BLOCK_X;

  // 每个 thread 负责 Tm * Tn 个元素计算
  constexpr int Tm = Bm / C_BLOCK_Y;
  constexpr int Tn = Bn / C_BLOCK_X;
  float Ct[Tm][Tn] = {0.0};

  // K- Loop
  for (int k = 0; k < K; k += Bk) {
    /* ------ 读取 global memory，存入 shared memory ------ */
    // 使用跨步循环，行方向的 stride 为 A_BLOCK_Y, 列方向的 stride 为 A_BLOCK_X
#pragma unroll
    for (int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y) {
      int r = r0 + i;
#pragma unroll
      for (int j = A_THREAD_X; j < Bk; j += A_BLOCK_X) {
        int c = k + j;
        As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
      }
    }

    // 使用跨步循环，行方向的 stride 为 B_BLOCK_Y, 列方向的 stride 为 B_BLOCK_X
#pragma unroll
    for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
      int r = k + i;
#pragma unroll
      for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
        int c = c0 + j;

        Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
      }
    } __syncthreads();

    /* ------ 计算 tileA * tileB ------ */
    // 先循环 k 维度，按向量外积的方式计算
#pragma unroll
    for (int p = 0; p < Bk; ++p) {
      // 使用跨步循环，行方向的 stride 为 C_BLOCK_Y, 列方向的 stride 为 C_BLOCK_X
#pragma unroll
      for (int i = 0; i < Tm; ++i) {
    
       int r = C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
        for (int j = 0; j < Tn; ++j) {
          int c = C_THREAD_X + j * C_BLOCK_X;
          Ct[i][j] += As[r][p] * Bs[p][c];
        }
      }
    }

    __syncthreads();
  }

  // 将 Ct 写入 C
#pragma unroll
  for (int i = 0; i < Tm; ++i) {
    int r = r0 + C_THREAD_Y + i * C_BLOCK_Y;
#pragma unroll
    for (int j = 0; j < Tn; ++j) {
      int c = c0 + C_THREAD_X + j * C_BLOCK_X;

      if (r < M && c < N) { C[r * N + c] = Ct[i][j]; }
    }
  }
}



// 矩阵初始化函数：填充随机浮点数 [0, 1]
void init_matrix_rand(float* mat, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < (size / sizeof(float)); ++i) {
        mat[i] = dis(gen);
    }
}

bool verify_result(const float* host_res, const float* device_res, size_t size, float epsilon = 1e-2f) {
    int error_count = 0;
    const int max_errors_to_print = 10; // 限制报错输出数量，避免刷屏

    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(host_res[i] - device_res[i]);
        if (diff > epsilon) {
            if (error_count < max_errors_to_print) {
                std::cerr << "结果错误 [索引 " << i << "]: CPU=" << host_res[i] 
                          << ", GPU=" << device_res[i] << ", 误差=" << diff << std::endl;
            }
            error_count++;
        }
    }

    if (error_count > 0) {
        std::cout << "❌ 验证失败！总计错误点: " << error_count << " / " << size << std::endl;
        return false;
    } else {
        std::cout << "✅ 验证通过！计算结果一致。" << std::endl;
        return true;
    }
}

int main(int argc, char* argv[]) {
    // 1. 参数解析
    int M = 2048, N = 2048, K = 2048;
    if (argc > 1) M = std::stoi(argv[1]);
    if (argc > 2) N = std::stoi(argv[2]);
    if (argc > 3) K = std::stoi(argv[3]);

    // 2. 设备信息输出
    int devID = 0;
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "运行设备: " << deviceProps.name << std::endl;
    std::cout << "矩阵规模: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;

    // 3. 内存分配
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);      // CPU 结果
    float* h_C_gpu_naive = (float*)malloc(size_C);
    float* C_gpu_cuBLAS = (float*)malloc(size_C);
    float* C_gpu_Blocktile = (float*)malloc(size_C);

    // 4. 数据初始化
    init_matrix_rand(h_A, size_A);
    init_matrix_rand(h_B, size_B);
    std::memset(h_C, 0, size_C);

    // 串行计算并计时
    // std::cout << "正在启动 CPU 串行计算..." << std::endl;
    // auto start = std::chrono::high_resolution_clock::now();

    // serial_gemm(h_A, h_B, h_C, M, N, K);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<float, std::milli> duration = end - start;
    // std::cout << "CPU 计算完成，耗时: " << duration.count() << " ms" << std::endl;


    //分配GPU内存
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void**)&d_A, size_A));
    checkCudaErrors(cudaMalloc((void**)&d_B, size_B));
    checkCudaErrors(cudaMalloc((void**)&d_C, size_C));

    checkCudaErrors(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);


    cudaEvent_t cuda_start, cuda_stop;
    checkCudaErrors(cudaEventCreate(&cuda_start));
    checkCudaErrors(cudaEventCreate(&cuda_stop));


/*                                 naive_gemm实现                                        */
    // 1. --- 重要：热身跑 (Warm-up) ---
    // 确保所有核函数代码已加载，硬件进入高频状态
    naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    checkCudaErrors(cudaDeviceSynchronize()); // 强制等待热身结束

    // 2. --- 循环计时 (10次) ---
    const int iters = 10;
    checkCudaErrors(cudaEventRecord(cuda_start));

    for (int i = 0; i < iters; i++) {
        naive_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    checkCudaErrors(cudaEventRecord(cuda_stop));
    checkCudaErrors(cudaEventSynchronize(cuda_stop)); // 确保 10 次全部算完

    float total_gpu_duration = 0;
    checkCudaErrors(cudaEventElapsedTime(&total_gpu_duration, cuda_start, cuda_stop));

    // 3. --- 计算平均值与性能 ---
    float avg_duration = total_gpu_duration / iters;
    double ops = 2.0 * M * N * K;
    double tflops = (ops * 1e-12) / (avg_duration * 1e-3); // 每秒万亿次运算

    std::cout << "Naive GPU 平均耗时: " << avg_duration << " ms" << std::endl;
    std::cout << "Naive GPU 吞吐量: " << tflops << " TFLOPS" << std::endl;

    // 4. 结果回传 (只需回传最后一次的结果用于验证)
    checkCudaErrors(cudaMemcpy(h_C_gpu_naive, d_C, size_C, cudaMemcpyDeviceToHost));
/*                                 naive_gemm实现                                        */



/*                                 cuBLAS实现                                        */

    // 1. 提前创建句柄
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta  = 0.0f;

    // 2. --- 重要：热身跑 (Warm-up) ---
    // 这一步不计时，目的是让 GPU 载入内核并分配好缓存
    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                                &alpha, d_B, N, d_A, K, &beta, d_C, N));
    cudaDeviceSynchronize(); // 确保热身彻底完成

    // 3. --- 开始计时测试 (取10次平均值) ---
    checkCudaErrors(cudaEventRecord(cuda_start));

    for (int i = 0; i < iters; i++) {
        checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                                    &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }

    checkCudaErrors(cudaEventRecord(cuda_stop));
    checkCudaErrors(cudaEventSynchronize(cuda_stop));

    float total_time = 0;
    checkCudaErrors(cudaEventElapsedTime(&total_time, cuda_start, cuda_stop));
    avg_duration = total_time / iters; // 算平均值

    // 4. 计算吞吐量 (TFLOPS)
    // 矩阵乘法运算量公式：2 * M * N * K
    ops = 2.0 * M * N * K;
    tflops = (ops * 1e-12) / (avg_duration * 1e-3); // 每秒万亿次运算

    std::cout << "cuBLAS 平均耗时: " << avg_duration << " ms" << std::endl;
    std::cout << "cuBLAS 吞吐量: " << tflops << " TFLOPS" << std::endl;

    cublasDestroy(handle);

    // 结果校验
    checkCudaErrors(cudaMemcpy(C_gpu_cuBLAS, d_C, size_C, cudaMemcpyDeviceToHost));

/*                                 cuBLAS实现                                        */




    const int Bm = 128;
    const int Bn = 128;
/*                                 blockTileGEMM实现                                        */

    block = 256;
    grid = dim3(CEIL(N, Bn), CEIL(M, Bm));
    
    blockTileGEMM<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    checkCudaErrors(cudaDeviceSynchronize()); // 强制等待热身结束

    // 2. --- 循环计时 (10次) ---
    checkCudaErrors(cudaEventRecord(cuda_start));

    for (int i = 0; i < iters; i++) {
        blockTileGEMM<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }

    checkCudaErrors(cudaEventRecord(cuda_stop));
    checkCudaErrors(cudaEventSynchronize(cuda_stop)); // 确保 10 次全部算完

    total_gpu_duration = 0;
    checkCudaErrors(cudaEventElapsedTime(&total_gpu_duration, cuda_start, cuda_stop));

    // 3. --- 计算平均值与性能 ---
    avg_duration = total_gpu_duration / iters;
    ops = 2.0 * M * N * K;
    tflops = (ops * 1e-12) / (avg_duration * 1e-3); // 每秒万亿次运算

    std::cout << "blockTileGEMM GPU 平均耗时: " << avg_duration << " ms" << std::endl;
    std::cout << "blockTileGEMM GPU 吞吐量: " << tflops << " TFLOPS" << std::endl;

    // 4. 结果回传 (只需回传最后一次的结果用于验证)
    checkCudaErrors(cudaMemcpy(C_gpu_Blocktile, d_C, size_C, cudaMemcpyDeviceToHost));




/*                                 blockTileGEMM实现                                        */


    // 结果验证
    verify_result(C_gpu_Blocktile, h_C_gpu_naive, size_C / sizeof(float));

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(cuda_start));
    checkCudaErrors(cudaEventDestroy(cuda_stop));

    return 0;
}