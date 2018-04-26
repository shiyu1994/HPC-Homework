default: sparse

CC = mpiicpc
CXX = mpigxx
CFLAGS = -O3 -qopenmp -mkl
CXXFLAGS = -O3 -fopenmp -std=c++11 -mavx2

sparse: main.cpp func.cpp check_func.cpp func.h sparse_ma.o sparse_mm.o
	$(CC) $(CFLAGS) $(CLIBS) main.cpp func.cpp check_func.cpp -o sparse sparse_ma.o sparse_mm.o

sparse_ma.o: sparse_ma.cpp sparse_ma.h
	$(CXX) $(CXXFLAGS) sparse_ma.cpp -c -o sparse_ma.o

sparse_mm.o: sparse_mm.cpp sparse_mm.h
	$(CXX) $(CXXFLAGS) sparse_mm.cpp -c -o sparse_mm.o

run: sparse
	#mpirun -np 1 ./sparse /apps/hw3/matrix/torso3.csr
	./sparse /apps/hw3/matrix/torso3.csr
	#srun -n 1 ./sparse /apps/hw3/matrix/torso3.csr

mrun: sparse
	numactl -m 1 ./sparse /apps/hw3/matrix/torso3.csr
clean:
	rm -rf ./sparse
	rm *.o
