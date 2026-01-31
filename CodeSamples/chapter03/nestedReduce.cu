#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 简单的错误检查宏
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                         \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));      \
        exit(1);                                                               \
    }                                                                          \
}

// 计时函数
double seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

// CPU 递归归约（作为验证基准）
int cpuRecursiveReduce(int *data, int const size) {
    if (size == 1) return data[0];
    int const stride = size / 2;
    for (int i = 0; i < stride; i++) data[i] += data[i + stride];
    return cpuRecursiveReduce(data, stride);
}

// 基础版：相邻配对归约（用于对比性能）
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if ((blockIdx.x * blockDim.x + tid) >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

/**
 * 动态并行版：递归归约
 * 注意：在 sm_90/sm_120 架构中，移除了设备端 cudaDeviceSynchronize
 */
__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize) {
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    // 递归终止条件
    if (isize == 2 && tid == 0) {
        *odata = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;
    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid + istride];
    }

    // 块内同步，确保这一层的加法完成
    __syncthreads();

    // 嵌套调用：由 0 号线程发起下一级子内核
    if (tid == 0 && istride > 1) {
        // 在 sm_120 上，启动后直接退出，隐式同步会处理层级关系
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        
        // 注意：此处不再调用 cudaDeviceSynchronize()
    }
}

int main(int argc, char **argv) {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nblock = 2048;
    int nthread = 512;
    if (argc > 1) nblock = atoi(argv[1]);
    if (argc > 2) nthread = atoi(argv[2]);

    int size = nblock * nthread;
    printf("Reduction configuration: array %d grid %d block %d\n", size, nblock, nthread);

    size_t bytes = size * sizeof(int);
    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(nblock * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    for (int i = 0; i < size; i++) h_idata[i] = 1;
    memcpy(tmp, h_idata, bytes);

    int *d_idata, *d_odata;
    CHECK(cudaMalloc((void **)&d_idata, bytes));
    CHECK(cudaMalloc((void **)&d_odata, nblock * sizeof(int)));

    // CPU 验证
    double iStart = seconds();
    int cpu_sum = cpuRecursiveReduce(tmp, size);
    double iElaps = seconds() - iStart;
    printf("CPU reduce elapsed: %f sec, sum: %d\n", iElaps, cpu_sum);

    // GPU 相邻配对
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    reduceNeighbored<<<nblock, nthread>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, nblock * sizeof(int), cudaMemcpyDeviceToHost));
    int gpu_sum = 0;
    for (int i = 0; i < nblock; i++) gpu_sum += h_odata[i];
    printf("GPU neighbored elapsed: %f sec, sum: %d\n", iElaps, gpu_sum);

    // GPU 动态并行递归
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    // 启动初始父内核
    gpuRecursiveReduce<<<nblock, nthread>>>(d_idata, d_odata, nthread);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, nblock * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i < nblock; i++) gpu_sum += h_odata[i];
    printf("GPU nested recursive elapsed: %f sec, sum: %d\n", iElaps, gpu_sum);

    free(h_idata); free(h_odata); free(tmp);
    CHECK(cudaFree(d_idata)); CHECK(cudaFree(d_odata));
    CHECK(cudaDeviceReset());

    return 0;
}