#include <stdio.h>
#include <cuda_runtime.h>

// 模拟计算任务
__global__ void simple_kernel(int iterations) {
    double val = threadIdx.x;
    for (int i = 0; i < iterations; i++) {
        val = sqrt(val + 5.0) * tan(0.1);
    }
}

int main() {
    // 1. 定义流与事件句柄
    cudaStream_t stream_A, stream_B;
    cudaEvent_t start_event, stop_event, sync_event;

    // 2. 初始化流
    cudaStreamCreate(&stream_A);
    cudaStreamCreate(&stream_B);

    // 3. 初始化事件
    // 使用 cudaEventDefault 或用 cudaEventDisableTiming 优化非计时事件
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming); // 仅用于同步，降低开销

    printf("开始执行流间同步与计时实验...\n");

    // 4. 记录起始时间戳
    cudaEventRecord(start_event, stream_A);

    // 在流 A 中启动预处理内核
    simple_kernel<<<128, 512, 0, stream_A>>>(100000); 

    // 5. 设置同步点：在流 A 中记录同步事件
    // 只有当流 A 的任务执行到这里，sync_event 才会变为“完成”状态
    cudaEventRecord(sync_event, stream_A);

    // 6. 流间等待：让流 B 等待流 A 的信号
    // 这是硬件级的等待，不阻塞主机 CPU
    cudaStreamWaitEvent(stream_B, sync_event, 0); 

    // 流 B 在接收到 sync_event 信号后才会启动后续内核
    simple_kernel<<<128, 512, 0, stream_B>>>(200000);

    // 7. 记录结束时间戳
    // 注意：记录在流 B 中，以测算整个链条的完成时间
    cudaEventRecord(stop_event, stream_B);

    // 8. 主机同步：等待 GPU 完成所有操作
    cudaEventSynchronize(stop_event);

    // 9. 计算并输出耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_event, stop_event); //

    printf("全流程 GPU 硬件耗时: %.4f ms\n", milliseconds);

    // 10. 清理资源
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(sync_event);
    cudaStreamDestroy(stream_A);
    cudaStreamDestroy(stream_B);

    return 0;
}