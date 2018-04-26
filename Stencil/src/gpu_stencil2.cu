#include "stencil.hpp"
#include <cuda.h>
#include <vector>
#include "aligned_allocator.h"
#include <chrono>
#include <iostream>

using std::chrono::system_clock;
using std::chrono::duration;
using std::endl;
using std::cout;

#define BLOCK_DIM_X 32
#define BLOCK_DIM_Y 32

GPUStencil::GPUStencil(const Config &configg, dvec64 &stencil, int block_x, int block_y, int block_t):
Stencil(configg, stencil, block_x, block_y, block_t) { 
    cuda_stencil1 = nullptr;
    cuda_stencil2 = nullptr;
    grid_dim_x = -1;
    grid_dim_y = -1;
    block_dim_x = -1;
    block_dim_y = -1;
}

void GPUStencil::AllocGPUMem() {
    size_t cuda_stencil_size = sizeof(double) * (x_dim + 2) * (y_dim + 2) * (z_dim + 2);
    cudaMalloc(&cuda_stencil1, cuda_stencil_size);
    cudaMalloc(&cuda_stencil2, cuda_stencil_size);
    cudaMemset(cuda_stencil1, 0, cuda_stencil_size);
    cudaMemset(cuda_stencil2, 0, cuda_stencil_size);
}

void GPUStencil::CopyDataFromCPUToGPU() {
    size_t cuda_stencil_size = sizeof(double) * (x_dim + 2) * (y_dim + 2) * (z_dim + 2);
    cudaMemcpy(cuda_stencil1, stencil.data(), cuda_stencil_size, cudaMemcpyHostToDevice);
}

void GPUStencil::CalcBlockAndGridDim() {
    block_dim_x = BLOCK_DIM_X;
    block_dim_y = BLOCK_DIM_Y / 4;

    grid_dim_x = (x_dim + block_dim_x - 1) / (block_dim_x - 2);
    grid_dim_y = (y_dim + block_dim_y - 1) / (block_dim_y - 2);
}

void GPUStencil::CopyResultFromGPUToCPU() {
    size_t cuda_stencil_size = sizeof(double) * (x_dim + 2) * (y_dim + 2) * (z_dim + 2);
    if(t_steps % 2 == 1) {
        cudaMemcpy(stencil.data(), cuda_stencil2, cuda_stencil_size, cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpy(stencil.data(), cuda_stencil1, cuda_stencil_size, cudaMemcpyDeviceToHost);
    }
}

void GPUStencil::FreeGPUMem() {
    cudaFree(cuda_stencil1);
    cudaFree(cuda_stencil2);
}

//2D blocking
__global__ void GPUStencilKernel(double *cuda_stencil1, double *cuda_stencil2, int dim_x, int dim_y, int dim_z, int t_steps, 
        double alpha, double beta_x_0, double beta_x_1, double beta_y_0, double beta_y_1, double beta_z_0, double beta_z_1) {
    double *cur_stencil = cuda_stencil1;
    int x_index = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int y_index = blockIdx.y * (4 * blockDim.y - 2) + threadIdx.y * 4;
    int threadIdxy = 4 * threadIdx.y;
    int z_size = (dim_x + 2) * (dim_y + 2);
    int y_size = dim_x + 2;
    __shared__ double subplanes[3][BLOCK_DIM_X][BLOCK_DIM_Y];

    for(int i = 0; i < t_steps; ++i) {
        int index = y_index * y_size + x_index;
        subplanes[0][threadIdxy][threadIdx.x] = cuda_stencil1[index];
        subplanes[0][threadIdxy + 1][threadIdx.x] = cuda_stencil1[index + y_size];
        subplanes[0][threadIdxy + 2][threadIdx.x] = cuda_stencil1[index + 2 * y_size];
        subplanes[0][threadIdxy + 3][threadIdx.x] = cuda_stencil1[index + 3 * y_size];
        index += z_size;
        subplanes[1][threadIdxy][threadIdx.x] = cuda_stencil1[index];
        subplanes[1][threadIdxy + 1][threadIdx.x] = cuda_stencil1[index + y_size];
        subplanes[1][threadIdxy + 2][threadIdx.x] = cuda_stencil1[index + 2 * y_size];
        subplanes[1][threadIdxy + 3][threadIdx.x] = cuda_stencil1[index + 3 * y_size];
        for(int j = 1; j < dim_z + 1; ++j) {
            //load into shared memory
            subplanes[(j + 1) % 3][threadIdxy][threadIdx.x] = cuda_stencil1[index + z_size]; 
            subplanes[(j + 1) % 3][threadIdxy + 1][threadIdx.x] = cuda_stencil1[index + z_size + y_size];
            subplanes[(j + 1) % 3][threadIdxy + 2][threadIdx.x] = cuda_stencil1[index + z_size + 2 * y_size];
            subplanes[(j + 1) % 3][threadIdxy + 3][threadIdx.x] = cuda_stencil1[index + z_size + 3 * y_size];
            __syncthreads();
            if(threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 &&
                threadIdxy > 0 && threadIdxy < 4 * blockDim.y - 1 &&
                x_index < dim_x + 1 && y_index < dim_y + 1) {
                cuda_stencil2[index] = alpha * subplanes[j % 3][threadIdxy][threadIdx.x] + 
                    beta_x_0 * subplanes[j % 3][threadIdxy][threadIdx.x - 1] + beta_x_1 * subplanes[j % 3][threadIdxy][threadIdx.x + 1] +
                    beta_y_0 * subplanes[j % 3][threadIdxy - 1][threadIdx.x] + beta_y_1 * subplanes[j % 3][threadIdxy + 1][threadIdx.x] + 
                    beta_z_0 * subplanes[(j + 2) % 3][threadIdxy][threadIdx.x] + beta_z_1 * subplanes[(j + 1) % 3][threadIdxy][threadIdx.x];
            } 
            if(threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 &&
                threadIdxy + 1 > 0 && threadIdxy + 1 < 4 * blockDim.y - 1 &&
                x_index < dim_x + 1 && y_index + 1 < dim_y + 1) {
                cuda_stencil2[index + y_size] = alpha * subplanes[j % 3][threadIdxy + 1][threadIdx.x] + 
                    beta_x_0 * subplanes[j % 3][threadIdxy + 1][threadIdx.x - 1] + beta_x_1 * subplanes[j % 3][threadIdxy + 1][threadIdx.x + 1] +
                    beta_y_0 * subplanes[j % 3][threadIdxy][threadIdx.x] + beta_y_1 * subplanes[j % 3][threadIdxy + 2][threadIdx.x] + 
                    beta_z_0 * subplanes[(j + 2) % 3][threadIdxy + 1][threadIdx.x] + beta_z_1 * subplanes[(j + 1) % 3][threadIdxy + 1][threadIdx.x];
            }
            if(threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 &&
                threadIdxy + 2 > 0 && threadIdxy + 2 < 4 * blockDim.y - 1 &&
                x_index < dim_x + 1 && y_index + 1 < dim_y + 1) {
                cuda_stencil2[index + 2 * y_size] = alpha * subplanes[j % 3][threadIdxy + 2][threadIdx.x] + 
                    beta_x_0 * subplanes[j % 3][threadIdxy + 2][threadIdx.x - 1] + beta_x_1 * subplanes[j % 3][threadIdxy + 2][threadIdx.x + 1] +
                    beta_y_0 * subplanes[j % 3][threadIdxy + 1][threadIdx.x] + beta_y_1 * subplanes[j % 3][threadIdxy + 3][threadIdx.x] + 
                    beta_z_0 * subplanes[(j + 2) % 3][threadIdxy + 2][threadIdx.x] + beta_z_1 * subplanes[(j + 1) % 3][threadIdxy + 2][threadIdx.x];
            }
            if(threadIdx.x > 0 && threadIdx.x < blockDim.x - 1 &&
                threadIdxy + 3 > 0 && threadIdxy + 3 < 4 * blockDim.y - 1 &&
                x_index < dim_x + 1 && y_index + 1 < dim_y + 1) {
                cuda_stencil2[index + 3 * y_size] = alpha * subplanes[j % 3][threadIdxy + 3][threadIdx.x] + 
                    beta_x_0 * subplanes[j % 3][threadIdxy + 3][threadIdx.x - 1] + beta_x_1 * subplanes[j % 3][threadIdxy + 3][threadIdx.x + 1] +
                    beta_y_0 * subplanes[j % 3][threadIdxy + 2][threadIdx.x] + beta_y_1 * subplanes[j % 3][threadIdxy + 4][threadIdx.x] + 
                    beta_z_0 * subplanes[(j + 2) % 3][threadIdxy + 3][threadIdx.x] + beta_z_1 * subplanes[(j + 1) % 3][threadIdxy + 3][threadIdx.x];
            }
            index += z_size;  
        }            
        __syncthreads();    
        //exchange
        cur_stencil = cuda_stencil2;
        cuda_stencil2 = cuda_stencil1;
        cuda_stencil1 = cur_stencil; 
    }
}

void GPUStencil::Compute() {
    cout << "allocating gpu memory" << endl;
    auto t_start = system_clock::now(); 
    AllocGPUMem();
    auto t_end = system_clock::now();
    duration<double> t_duration = t_end - t_start;
    cout << "allocating gpu memory time: " << t_duration.count() << endl;

    cout << "copy data to gpu" << endl;
    t_start = system_clock::now();
    CopyDataFromCPUToGPU();
    t_end = system_clock::now();
    t_duration = t_end - t_start;
    cout << "copy data to gpu time: " << t_duration.count() << endl;

  
    CalcBlockAndGridDim();
      

    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(grid_dim_x, grid_dim_y);

    cout << "gpu kernel start" << endl;
    t_start = system_clock::now();  
    GPUStencilKernel<<<dimGrid, dimBlock>>>(cuda_stencil1, cuda_stencil2, x_dim, y_dim, z_dim, t_steps, alpha, beta_x_0, beta_x_1,
        beta_y_0, beta_y_1, beta_z_0, beta_z_1);
    cudaDeviceSynchronize();
    t_end = system_clock::now();
    t_duration = t_end - t_start;
    cout << "gpu kernel time: " << t_duration.count() << endl;

    cout << "copy result from gpu" << endl;
    t_start = system_clock::now();
    CopyResultFromGPUToCPU();
    t_end = system_clock::now();
    t_duration = t_end - t_start;
    cout << "copy result from gpu time: " << t_duration.count() << endl;

    FreeGPUMem();
}
