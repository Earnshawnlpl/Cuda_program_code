#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // CPU 初始化
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // 启动核函数
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize>>>(a, b, c, N);

    cudaDeviceSynchronize();

    printf("Result: %f\n", c[0]);  // 自动触发回迁（如有必要）

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
