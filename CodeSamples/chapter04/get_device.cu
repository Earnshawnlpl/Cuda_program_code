#include <stdio.h>
#include <cuda_runtime.h>

// 定义错误检查宏，确保每个 CUDA 调用都成功
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

int main(int argc, char **argv)
{
    printf("--- CUDA Device Physical Limits Query ---\n");

    int deviceCount = 0;
    CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("No CUDA supporting devices found.\n");
        return 1;
    }

    int dev = 0; // 查询第一个设备
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    printf("\n--- Core Resource Limits (Per Block) ---\n");
    // 每个线程块可用的最大 32 位寄存器数量
    printf("Max registers per block:         %d\n", prop.regsPerBlock);
    // 每个线程块可用的最大共享内存 (静态分配)
    printf("Max shared memory per block:     %zu KB\n", prop.sharedMemPerBlock / 1024);
    // 每个线程块允许的最大线程数
    printf("Max threads per block:           %d\n", prop.maxThreadsPerBlock);
    // 线程块在 X, Y, Z 三个维度的最大尺寸
    printf("Max block dimensions:            (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    printf("\n--- SM (Streaming Multiprocessor) Limits ---\n");
    // 该 GPU 拥有的 SM 数量
    printf("Number of SMs:                   %d\n", prop.multiProcessorCount);
    // 每个 SM 最大活跃线程数
    printf("Max threads per SM:              %d\n", prop.maxThreadsPerMultiProcessor);
    // 每个 SM 最大活跃线程束 (Warps)
    printf("Max warps per SM:                %d\n", prop.maxThreadsPerMultiProcessor / 32);
    // 每个 SM 最大共享内存
    printf("Max shared memory per SM:        %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);

    printf("\n--- Global Memory (DRAM) ---\n");
    printf("Total Global Memory:             %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("Memory Bus Width:                %d-bit\n", prop.memoryBusWidth);
    
    printf("------------------------------------------\n");

    return 0;
}