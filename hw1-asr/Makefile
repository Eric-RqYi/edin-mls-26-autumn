# Build the kernels and the tests. Run on a GPU node.
#
#   make            build build/sm_XX/libasr_kernels.so and the four tests
#   make lib        build only the library (what ./run.sh needs)
#   make test       build, then run all four tests
#   make test1      build, then run only the task 1 test (also test2, test3, test4)
#   make clean      delete build/
#
# XX is the compute capability of the GPU you are on (75 = RTX 2080 Ti,
# 86 = RTX A6000, 90 = H200), so builds for different GPUs do not mix.
# On a machine without a GPU, say which GPU to build for: make SM=75

NVCC ?= nvcc
SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.')
SRC ?= csrc
BUILD ?= build/sm_$(SM)

CUDA_HOME := $(abspath $(dir $(shell which $(NVCC)))/..)
# -diag-suppress 177: no "declared but never referenced" warnings for blanks you have not filled yet
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -diag-suppress 177
FLAGS := $(NVCCFLAGS) -arch=sm_$(SM) -Xcompiler -fPIC -I$(SRC)

KERNEL_SRCS := $(SRC)/task1_rope.cu $(SRC)/task2_reduce.cu $(SRC)/task3_gemm.cu \
               $(SRC)/task4_attention.cu $(SRC)/provided.cu
HEADERS := $(wildcard $(SRC)/*.cuh) $(SRC)/asr_kernels.h
OBJS := $(patsubst $(SRC)/%.cu,$(BUILD)/obj/%.o,$(KERNEL_SRCS))
LIB := $(BUILD)/libasr_kernels.so
TESTS := $(BUILD)/test_task1_rope $(BUILD)/test_task2_reduce \
         $(BUILD)/test_task3_gemm $(BUILD)/test_task4_attention

all: check-sm $(LIB) $(TESTS)

lib: check-sm $(LIB)

check-sm:
ifeq ($(strip $(SM)),)
	$(error No GPU found. Build on a GPU node (srun ...), or name the GPU: make SM=75)
endif

$(BUILD)/obj/%.o: $(SRC)/%.cu $(HEADERS)
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) -c $< -o $@

$(LIB): $(OBJS)
	$(NVCC) $(FLAGS) -shared $^ -o $@

# Tests link the kernel objects directly. The GEMM test also uses cuBLAS as a reference.
$(BUILD)/test_%: tests/test_%.cu tests/test_util.h $(OBJS)
	$(NVCC) $(FLAGS) $< $(OBJS) -o $@ -lcublas -Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64

test: all
	@status=0; for t in $(TESTS); do $$t || status=1; done; exit $$status

test1: check-sm $(BUILD)/test_task1_rope
	$(BUILD)/test_task1_rope
test2: check-sm $(BUILD)/test_task2_reduce
	$(BUILD)/test_task2_reduce
test3: check-sm $(BUILD)/test_task3_gemm
	$(BUILD)/test_task3_gemm
test4: check-sm $(BUILD)/test_task4_attention
	$(BUILD)/test_task4_attention

clean:
	rm -rf build

.PHONY: all lib check-sm test test1 test2 test3 test4 clean
.SECONDARY: $(OBJS)
