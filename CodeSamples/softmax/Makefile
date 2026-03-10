# 1. 指定 CUDA 编译器
NVCC = nvcc

# 2. 设置目标文件名
TARGET = softmax_exe

# 3. 设置 GPU 架构 (非常重要！)
ARCH = -arch=sm_120

# 4. 编译选项
# -O3: 开启最高级别优化
# -Xcompiler: 将后面的参数传递给底层的 GCC 编译器 (如 -Wall)
NVCCFLAGS = -O3 $(ARCH) -Xcompiler -Wall

# 5. 链接库
# -lm: 数学库
# -lcublas: 如果你用了 cuBLAS (你之前的头文件里有)
# -lcudart: CUDA Runtime 库
LIBS = -lm -lcublas -lcudart

# --- 编译规则 ---

all: $(TARGET)

$(TARGET): softmax.cu
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) softmax.cu $(LIBS)

# 运行程序
run: all
	./$(TARGET)

# 清理文件
clean:
	rm -f $(TARGET)