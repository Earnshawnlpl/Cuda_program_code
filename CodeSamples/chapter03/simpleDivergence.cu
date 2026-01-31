#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * simpleDivergence demonstrates divergent code on the GPU and its impact on
 * performance and CUDA metrics.
 */

__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        ia = 100.0f;
    }

    if (!ipred)
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if ((itid & 0x01) == 0) // 注意这里加了括号，保证位运算优先级正确
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

// mathKernel5: 线程级严重分化 (奇偶线程走不同循环)
__global__ void mathKernel5(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    // 分支 1：线程级分化
    if (tid % 2 == 0) 
    {
        // 路径 A：复杂的循环计算
        for (int i = 0; i < 500; ++i) {
            val += sqrtf(val + (float)i);
        }
    } 
    else 
    {
        // 路径 B：同样复杂的不同计算
        for (int i = 0; i < 500; ++i) {
            val += fabsf(val - (float)i);
        }
    }

    c[tid] = val;
}

// mathKernel6: Warp 级对齐 (整个 Warp 走相同循环)
__global__ void mathKernel6(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    // 分支 2：Warp 级分化 (无 Warp 内部冲突)
    if ((tid / 32) % 2 == 0) 
    {
        for (int i = 0; i < 500; ++i) {
            val += sqrtf(val + (float)i);
        }
    } 
    else 
    {
        for (int i = 0; i < 500; ++i) {
            val += fabsf(val - (float)i);
        }
    }

    c[tid] = val;
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 1 << 24;
    int blocksize = 512;

    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size      = atoi(argv[2]);

    printf("Data size %d ", size);

    // set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // 修改点 1：将计时变量改为 double
    double iStart, iElaps;

    // run a warmup kernel
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    // 修改点 2：printf 使用 %f 打印 double
    printf("warmup      <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // run kernel 1
    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // run kernel 2
    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // run kernel 3
    iStart = seconds();
    mathKernel3<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel3 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // run kernel 4
    iStart = seconds();
    mathKernel4<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("mathKernel4 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // mathKernel5 运行
    iStart = seconds();
    mathKernel5<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel5 (Divergent Loop) elapsed %f sec\n", iElaps);

    // mathKernel6 运行
    iStart = seconds();
    mathKernel6<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel6 (Aligned Loop)   elapsed %f sec\n", iElaps);

    // free gpu memory
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}