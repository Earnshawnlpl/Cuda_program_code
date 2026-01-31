#include <stdio.h>
#include <cuda_runtime.h>

// 一个简单的耗时核函数，用于模拟繁重的计算任务
__global__ void heavy_compute_kernel(int iterations) {
    double val = 0.0;
    for (int i = 0; i < iterations; i++) {
        val = sin(val) + cos(val);
    }
}

int main() {
    int priority_low, priority_high;
    
    // 1. 获取当前设备的优先级范围
    // 数值越小，优先级越高
    cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
    printf("设备支持的优先级范围: 最低=%d, 最高=%d\n", priority_low, priority_high);

    // 2. 创建不同优先级的非默认流
    cudaStream_t st_high, st_low;
    cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
    cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);

    // 创建计时事件
    cudaEvent_t start_h, stop_h, start_l, stop_l;
    cudaEventCreate(&start_h); cudaEventCreate(&stop_h);
    cudaEventCreate(&start_l); cudaEventCreate(&stop_l);

    printf("启动核函数并行竞争...\n");

    // 3. 在低优先级流中启动核函数 (先启动)
    // 根据描述，由于核函数启动是异步的，主机会立即返回并执行下一行
    cudaEventRecord(start_l, st_low);
    for(int i=0; i<10; i++) {
        heavy_compute_kernel<<<100, 1024, 0, st_low>>>(1000000); //
    }
    cudaEventRecord(stop_l, st_low);

    // 4. 在高优先级流中启动核函数 (后启动，尝试插队)
    cudaEventRecord(start_h, st_high);
    for(int i=0; i<10; i++) {
        heavy_compute_kernel<<<100, 1024, 0, st_high>>>(1000000); //
    }
    cudaEventRecord(stop_h, st_high);

    // 5. 等待所有流完成同步
    cudaStreamSynchronize(st_high);
    cudaStreamSynchronize(st_low);

    // 计算执行时间
    float ms_h, ms_l;
    cudaEventElapsedTime(&ms_h, start_h, stop_h);
    cudaEventElapsedTime(&ms_l, start_l, stop_l);

    printf("高优先级流耗时: %.2f ms\n", ms_h);
    printf("低优先级流耗时: %.2f ms\n", ms_l);

    if (ms_h < ms_l) {
        printf("验证成功：高优先级任务在硬件层面获得了优先调度。\n");
    }

    // 6. 销毁资源
    cudaStreamDestroy(st_high);
    cudaStreamDestroy(st_low);
    cudaEventDestroy(start_h); cudaEventDestroy(stop_h);
    cudaEventDestroy(start_l); cudaEventDestroy(stop_l);

    return 0;
}