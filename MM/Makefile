all: mm_cpu mm_gpu mm_knl mm_cpu_io

#compile for cpu
#CPU_LINK_FLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
CPU_FLAGS := -std=c++11 -fopenmp -O3 -mavx2 -mfma -Iinclude #-I${MKLROOT}/include

mm_cpu: obj/main_cpu.o obj/mm_cpu.o
	g++ $(CPU_FLAGS) -o mm_cpu obj/main_cpu.o obj/mm_cpu.o #$(CPU_LINK_FLAGS)
obj/main_cpu.o: src/cpu/main.cpp
	g++ $(CPU_FLAGS) src/cpu/main.cpp -c -o obj/main_cpu.o
obj/mm_cpu.o: src/cpu/matrix_multiplier.cpp
	g++ $(CPU_FLAGS) src/cpu/matrix_multiplier.cpp -c -o obj/mm_cpu.o




#compile for gpu
GPU_CXX_FLAGS := -Iinclude -std=c++11 -fopenmp -O3 #-I${MKLROOT}/include 
GPU_NVCC_FLAGS := -Iinclude -std=c++11 -O3 -arch=compute_32 
#GPU_LINK_FLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

mm_gpu: obj/main_gpu.o obj/mm_gpu.o
	g++ -O3 -fopenmp -L/usr/local/cuda/lib64 -lcudart -o mm_gpu obj/main_gpu.o obj/mm_gpu.o #$(GPU_LINK_FLAGS) 
obj/main_gpu.o: src/gpu/main.cpp
	g++ $(GPU_CXX_FLAGS) src/gpu/main.cpp -c -o obj/main_gpu.o
obj/mm_gpu.o: src/gpu/gpu_matrix_multiplier.cu
	nvcc $(GPU_NVCC_FLAGS) src/gpu/gpu_matrix_multiplier.cu -c -o obj/mm_gpu.o



#compile for knl
KNL_CXX_FLAGS := -fopenmp  -Iinclude -std=c++11 -xmic-avx512 -O3 #-I${MKLROOT}/include 
#KNL_LINK_FLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl

mm_knl: obj/main_knl.o obj/mm_knl.o
	icc $(KNL_CXX_FLAGS) -o mm_knl obj/main_knl.o obj/mm_knl.o #$(KNL_LINK_FLAGS)
obj/main_knl.o: src/knl/main.cpp
	icc $(KNL_CXX_FLAGS) src/knl/main.cpp -c -o obj/main_knl.o
obj/mm_knl.o: src/knl/knl_matrix_multiplier.cpp
	icc $(KNL_CXX_FLAGS) src/knl/knl_matrix_multiplier.cpp -c -o obj/mm_knl.o



#compile for cpu-io
#CPU_IO_LINK_FLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
CPU_IO_FLAGS := -std=c++11 -fopenmp -O3 -mavx2 -mfma  -Iinclude #-I${MKLROOT}/include 

mm_cpu_io: obj/main_cpu_io.o obj/mm_cpu_io.o
	g++ $(CPU_IO_FLAGS) -o mm_cpu_io obj/main_cpu_io.o obj/mm_cpu_io.o $(CPU_IO_LINK_FLAGS)
obj/main_cpu_io.o: src/cpu-io/main.cpp
	g++ $(CPU_IO_FLAGS) src/cpu-io/main.cpp -c -o obj/main_cpu_io.o
obj/mm_cpu_io.o: src/cpu-io/matrix_multiplier.cpp
	g++ $(CPU_IO_FLAGS) src/cpu-io/matrix_multiplier.cpp -c -o obj/mm_cpu_io.o




clean:
	rm mm_cpu mm_gpu mm_knl mm_cpu_io obj/*.o *.o io/*
