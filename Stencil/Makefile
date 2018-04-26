all: stencil_cpu stencil_gpu stencil_cpu_queue stencil_cpu_gpu stencil_knl

INCDIR := include

#compile for cpu
CPU_FLAGS := -O3 -fopenmp -I$(INCDIR) -std=c++11 -mavx2 
stencil_cpu: obj/main_cpu.o obj/stencil_cpu.o
	g++ $(CPU_FLAGS) -o stencil_cpu obj/main_cpu.o obj/stencil_cpu.o
obj/main_cpu.o: src/cpu/main.cpp
	g++ $(CPU_FLAGS) src/cpu/main.cpp -c -o obj/main_cpu.o
obj/stencil_cpu.o: src/cpu/stencil.cpp
	g++ $(CPU_FLAGS) src/cpu/stencil.cpp -c -o obj/stencil_cpu.o

#compile for knl
KNL_FLAGS := -O3 -fopenmp -I$(INCDIR) -std=c++11 -xmic-avx512
stencil_knl: obj/main_knl.o obj/stencil_knl.o
	icc $(KNL_FLAGS) -o stencil_knl obj/main_knl.o obj/stencil_knl.o
obj/main_knl.o: src/knl/main.cpp
	icc $(KNL_FLAGS) src/knl/main.cpp -c -o obj/main_knl.o
obj/stencil_knl.o: src/knl/stencil.cpp
	icc $(KNL_FLAGS) src/knl/stencil.cpp -c -o obj/stencil_knl.o

#compile for cpu-gpu
GPU_CPU_NVCC_FLAGS := -O3 -std=c++11 -arch=compute_32 -I${INCDIR}
GPU_CPU_GXX_FLAGS := -O3 -fopenmp -I$(INCDIR) -std=c++11 -mavx2 -L/usr/local/cuda/lib64 -lcudart 
stencil_cpu_gpu: obj/main_gpu_cpu.o obj/stencil_gpu_cpu.o obj/stencil_cpu_for_cpu_gpu.o obj/stencil_gpu_for_cpu_gpu.o
	g++ $(GPU_CPU_GXX_FLAGS) -o stencil_cpu_gpu obj/stencil_gpu_cpu.o obj/main_gpu_cpu.o obj/stencil_cpu_for_cpu_gpu.o obj/stencil_gpu_for_cpu_gpu.o
obj/stencil_gpu_cpu.o: src/cpu-gpu/gpu_cpu_stencil.cu
	nvcc $(GPU_CPU_NVCC_FLAGS) src/cpu-gpu/gpu_cpu_stencil.cu -c -o obj/stencil_gpu_cpu.o
obj/main_gpu_cpu.o: src/cpu-gpu/main.cpp
	g++ $(GPU_CPU_GXX_FLAGS) src/cpu-gpu/main.cpp -c -o obj/main_gpu_cpu.o
obj/stencil_cpu_for_cpu_gpu.o: src/cpu/stencil.cpp
	g++ $(GPU_CPU_GXX_FLAGS) src/cpu/stencil.cpp -c -o obj/stencil_cpu_for_cpu_gpu.o
obj/stencil_gpu_for_cpu_gpu.o: src/gpu/gpu_stencil.cu
	nvcc $(GPU_CPU_NVCC_FLAGS) src/gpu/gpu_stencil.cu -c -o obj/stencil_gpu_for_cpu_gpu.o


#compile for gpu
GPU_NVCC_FLAGS := -O3 -std=c++11 -arch=compute_32 -I${INCDIR}
GPU_GXX_FLAGS := -O3 -fopenmp -I$(INCDIR) -std=c++11 -mavx2 -L/usr/local/cuda/lib64 -lcudart 
stencil_gpu: obj/main_gpu.o obj/stencil_gpu.o obj/stencil_cpu_for_gpu.o
	g++ $(GPU_GXX_FLAGS) -o stencil_gpu obj/stencil_gpu.o obj/main_gpu.o obj/stencil_cpu_for_gpu.o
obj/stencil_gpu.o: src/gpu/gpu_stencil.cu
	nvcc $(GPU_NVCC_FLAGS) src/gpu/gpu_stencil.cu -c -o obj/stencil_gpu.o
obj/main_gpu.o: src/gpu/main.cpp
	g++ $(GPU_GXX_FLAGS) src/gpu/main.cpp -c -o obj/main_gpu.o
obj/stencil_cpu_for_gpu.o: src/cpu/stencil.cpp
	g++ $(GPU_GXX_FLAGS) src/cpu/stencil.cpp -c -o obj/stencil_cpu_for_gpu.o


#compile for cpu-queue
CPU_QUEUE_FLAGS := -O3 -fopenmp -I$(INCDIR) -std=c++11 -mavx2 
stencil_cpu_queue: obj/main_cpu_queue.o obj/stencil_cpu_queue.o obj/stencil_cpu_for_cpu_queue.o
	mpigxx $(CPU_QUEUE_FLAGS) -o stencil_cpu_queue obj/main_cpu_queue.o obj/stencil_cpu_queue.o obj/stencil_cpu_for_cpu_queue.o
obj/main_cpu_queue.o: src/cpu-queue/main.cpp
	mpigxx $(CPU_QUEUE_FLAGS) src/cpu-queue/main.cpp -c -o obj/main_cpu_queue.o
obj/stencil_cpu_queue.o: src/cpu-queue/cpu_queue_stencil.cpp
	mpigxx $(CPU_QUEUE_FLAGS) src/cpu-queue/cpu_queue_stencil.cpp -c -o obj/stencil_cpu_queue.o
obj/stencil_cpu_for_cpu_queue.o: src/cpu/stencil.cpp
	mpigxx $(CPU_QUEUE_FLAGS) src/cpu/stencil.cpp -c -o obj/stencil_cpu_for_cpu_queue.o



clean:
	rm stencil_cpu stencil_gpu stencil_cpu_gpu stencil_knl stencil_cpu_queue obj/*.o *.o
