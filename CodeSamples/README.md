

Copy from: www.wrox.com/go/procudac
Book: Professional CUDA C Progtamming
Follow by: gengyou.lu

# nvcc 命令参数说明
```
--ptxas-options=-v :获得每个线程的寄存器和每个块的共享内存资源的使用情况


```


# 第一章学习
```
# 代码清单1-1
nvcc hello.cu -o ../build/chapter01/hello
```

# 第二章学习
```
# 代码清单2-1
nvcc -Xcompiler -std=c99 sumArraysOnHost.c -o ../build/chapter02/sum
# 代码清单2-2
nvcc checkDimension.cu -o ../build/chapter02/checkDimension
# 代码清单2-3
nvcc defineGridBlock.cu -o ../build/chapter02/defineGridBlock
# 代码清单2-4
nvcc sumArraysOnGPU-small-case.cu -o ../build/chapter02/sumArraysOnGPU-small-case
# 代码清单2-5
nvcc sumArraysOnGPU-timer.cu -o ../build/chapter02/sumArraysOnGPU-timer
# 代码清单2-6
nvcc checkThreadIndex.cu -o ../build/chapter02/checkThreadIndex
# 代码清单2-7
nvcc sumMatrixOnGPU-2D-grid-2D-block.cu -o ../build/chapter02/sumMatrixOnGPU-2D-grid-2D-block
nvcc sumMatrixOnGPU-1D-grid-1D-block.cu -o ../build/chapter02/sumMatrixOnGPU-1D-grid-1D-block
nvcc sumMatrixOnGPU-2D-grid-1D-block.cu -o ../build/chapter02/sumMatrixOnGPU-2D-grid-1D-block
# 代码清单2-8
nvcc checkDeviceInfor.cu -o ../build/chapter02/checkDeviceInfor
```

# 第三章学习
```
# 代码清单3-1
nvcc simpleDivergence.cu -o ../build/chapter03/simpleDivergence
# 代码清单3-2
nvcc simpleDeviceQuery.cu -o ../build/chapter03/simpleDeviceQuery
```

# 第四章学习
```
# 代码清单4-1
nvcc globalVariable.cu -o ../build/chapter04/globalVariable


```

