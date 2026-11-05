CXX ?= g++
COMMON = -O3 -g -std=c++17 -fopenmp -march=native -Wall -Wextra -Wpedantic
BUILD = build

all: $(BUILD)/spmm_bench
$(BUILD):
	mkdir -p $(BUILD)
$(BUILD)/rows_scalar.o: src/rows_scalar.cpp src/kernels.h | $(BUILD)
	$(CXX) $(COMMON) -fno-tree-vectorize -fno-tree-slp-vectorize -c $< -o $@
$(BUILD)/rows_simd.o: src/rows_simd.cpp src/kernels.h | $(BUILD)
	$(CXX) $(COMMON) -fopt-info-vec-all=$(BUILD)/vectorization.txt -c $< -o $@
$(BUILD)/spmm_bench.o: src/spmm_bench.cpp src/kernels.h src/pmu.h | $(BUILD)
	$(CXX) $(COMMON) -c $< -o $@
$(BUILD)/spmm_bench: $(BUILD)/spmm_bench.o $(BUILD)/rows_scalar.o $(BUILD)/rows_simd.o
	$(CXX) $(COMMON) $^ -o $@
clean:
	rm -rf $(BUILD)
.PHONY: all clean
