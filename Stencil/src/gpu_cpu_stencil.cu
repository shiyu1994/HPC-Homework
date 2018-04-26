#include <cuda.h>
#include "stencil.hpp"
#include "aligned_allocator.h"
#include <chrono>

using std::chrono::system_clock;
using std::chrono::duration;
using std::endl;
using std::cout;

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_T 6

GPUCPUStencil::GPUCPUStencil(const Config &configg, dvec64 &stencil, int block_xx, int block_yy, int block_tt):
GPUStencil(configg, stencil, block_xx, block_yy, block_tt) {}


void GPUCPUStencil::CalcBlockAndGridDim(int block_t) { 
    block_dim_x = BLOCK_DIM_X;
    block_dim_y = BLOCK_DIM_Y;

    grid_dim_x = (x_dim + BLOCK_DIM_X - 2 * block_t - 1) / (BLOCK_DIM_X - 2 * block_t);
    grid_dim_y = (y_dim + BLOCK_DIM_Y - 2 * block_t - 1) / (BLOCK_DIM_Y - 2 * block_t);
}

//3.5D blocking, boundary is calculated by cpu
__global__ void GPUCPUStencilKernel(double *cuda_stencil1, double *cuda_stencil2, int dim_x, int dim_y, int dim_z, int block_t,
        double alpha, double beta_x_0, double beta_x_1, double beta_y_0, double beta_y_1, double beta_z_0, double beta_z_1) {
    //assert BLOCK_T == block_t
    __shared__ double subplanes[BLOCK_T][3][BLOCK_DIM_Y][BLOCK_DIM_X];  
    
    int z_size = (2 + dim_y) * (2 + dim_x);
    int y_size = (2 + dim_x);

    int x_index = blockIdx.x * (blockDim.x - 2 * block_t) + threadIdx.x;
    int y_index = blockIdx.y * (blockDim.y - 2 * block_t) + threadIdx.y; 

    for(int zz = 0; zz < dim_z + block_t + 1; ++zz) {
        int z = zz; 
        int index = z_size * z + y_size * y_index + x_index; 
        
        if(z < dim_z + 2 && x_index < dim_x + 2 && y_index < dim_y + 2) {            
            subplanes[0][z % 3][threadIdx.y][threadIdx.x] = cuda_stencil1[index]; 
        } 
        __syncthreads();
        for(int tt = 1; tt < block_t; ++tt) {
            z = zz - tt;
            if(z > 0 && z < dim_z + 1) { 
                if(threadIdx.x >= tt && threadIdx.x < BLOCK_DIM_X - tt &&
                    threadIdx.y >= tt && threadIdx.y < BLOCK_DIM_Y - tt && x_index < dim_x + 2 - tt && y_index < dim_y + 2 - tt) {
                    subplanes[tt][z % 3][threadIdx.y][threadIdx.x] = alpha * subplanes[tt - 1][z % 3][threadIdx.y][threadIdx.x] +
                        beta_z_0 * subplanes[tt - 1][(z + 2) % 3][threadIdx.y][threadIdx.x] + beta_z_1 * subplanes[tt - 1][(z + 1) % 3][threadIdx.y][threadIdx.x] +
                        beta_y_0 * subplanes[tt - 1][z % 3][threadIdx.y - 1][threadIdx.x] + beta_y_1 * subplanes[tt - 1][z % 3][threadIdx.y + 1][threadIdx.x] +
                        beta_x_0 * subplanes[tt - 1][z % 3][threadIdx.y][threadIdx.x - 1] + beta_x_1 * subplanes[tt - 1][z % 3][threadIdx.y][threadIdx.x + 1];
                }
            }
            else if(z == 0 || z == dim_z + 1) { 
                subplanes[tt][z % 3][threadIdx.y][threadIdx.x] = 0.0;
            } 
            __syncthreads();
        }
        z = zz - block_t; 
        if(z > 0 && z < dim_z + 1 && threadIdx.x >= block_t && threadIdx.x < BLOCK_DIM_X - block_t &&
                threadIdx.y >= block_t && threadIdx.y < BLOCK_DIM_Y - block_t && x_index < dim_x + 2 - block_t && y_index < dim_y + 2 - block_t) {
            index = z_size * z + y_size * y_index + x_index;
            cuda_stencil2[index] = alpha * subplanes[block_t - 1][z % 3][threadIdx.y][threadIdx.x] +
                beta_z_0 * subplanes[block_t - 1][(z + 2) % 3][threadIdx.y][threadIdx.x] + beta_z_1 * subplanes[block_t - 1][(z + 1) % 3][threadIdx.y][threadIdx.x] +
                beta_y_0 * subplanes[block_t - 1][z % 3][threadIdx.y - 1][threadIdx.x] + beta_y_1 * subplanes[block_t - 1][z % 3][threadIdx.y + 1][threadIdx.x] +
                beta_x_0 * subplanes[block_t - 1][z % 3][threadIdx.y][threadIdx.x - 1] + beta_x_1 * subplanes[block_t - 1][z % 3][threadIdx.y][threadIdx.x + 1];
        }
        __syncthreads();
    }
}

void GPUCPUStencil::CopyBoundaryFromGPU(double *host_stencil, double *cuda_stencil) {
    size_t size = (2 + x_dim) * (2 + y_dim) * (2 + z_dim) * 8;
    cudaMemcpy(host_stencil, cuda_stencil, size, cudaMemcpyDeviceToHost); 
}

void GPUCPUStencil::CopyBoundaryToGPU(double *host_stencil, double *cuda_stencil, int y_size, int z_size, int steps) { 
//#pragma omp parallel for schedule(static) num_threads(num_threads)
    /*for(int i = 1; i < z_dim + 1; ++i) {
        for(int j = 1; j < steps; ++j) {
            int offset = i * z_size + j * y_size + 1;
            size_t size = sizeof(double) * x_dim;
            cudaMemcpy(cuda_stencil + offset, host_stencil + offset, size, cudaMemcpyHostToDevice);
        }
        for(int j = steps; j < y_dim + 2 - steps; ++j) {
            int offset = i * z_size + j * y_size + 1;
            size_t size = sizeof(double) * (block_t - 1);
            cudaMemcpy(cuda_stencil + offset, host_stencil + offset, size, cudaMemcpyHostToDevice);
            offset = i * z_size + j * y_size + (x_dim - block_t + 2);
            cudaMemcpy(cuda_stencil + offset, host_stencil + offset, size, cudaMemcpyHostToDevice);
        }
        for(int j = y_dim + 2 - steps; j < y_dim + 1; ++j) { 
            int offset = i * z_size + j * y_size + 1;
            size_t size = sizeof(double) * x_dim;
            cudaMemcpy(cuda_stencil + offset, host_stencil + offset, size, cudaMemcpyHostToDevice);
        }
    }*/
    size_t size = (2 + x_dim) * (2 + y_dim) * (2 + z_dim) * 8;
    cudaMemcpy(cuda_stencil, host_stencil, size, cudaMemcpyHostToDevice); 

}

void GPUCPUStencil::CopyResultFromGPUToCPU(double *cuda_stencil, double *host_stencil, int block_t) {
    /*if(block_t >= 2) {
        int z_size = (2 + x_dim) * (2 + y_dim);
        int y_size = 2 + x_dim;
#pragma omp parallel for schedule(static) num_threads(num_threads)  
        for(int z = 1; z < z_dim + 1; ++z) {
            for(int y = block_t; y < y_dim + 2 - block_t; ++y) {
                int offset = z * z_size + y * y_size + block_t;
                cudaMemcpy(host_stencil + offset, cuda_stencil + offset, sizeof(double) * (x_dim - 2 * block_t + 2), cudaMemcpyDeviceToHost); 
            }
        }
    }
    else {*/
        cudaMemcpy(host_stencil, cuda_stencil, sizeof(double) * (2 + x_dim) * (2 + y_dim) * (2 + z_dim), cudaMemcpyDeviceToHost);
    //}
}

void GPUCPUStencil::CopyGPUResult(double *host_stencil1, double *host_stencil2, int steps) {
    int z_size = (2 + x_dim) * (2 + y_dim);
    int y_size = 2 + x_dim;
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int z = 1; z < z_dim + 1; ++z) {
        for(int y = steps; y < y_dim + 2 - steps; ++y) {
            for(int x = steps; x < x_dim + 2 - steps; ++x) {
                int index = z * z_size + y * y_size + x;
                host_stencil2[index] = host_stencil1[index];
            }
        }
    }
}

void GPUCPUStencil::Compute() {
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

  
    CalcBlockAndGridDim(block_t); 

    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(grid_dim_x, grid_dim_y);

    cout << "gpu kernel start" << endl;
    t_start = system_clock::now();  
    int z_size = (2 + x_dim) * (2 + y_dim);
    int y_size = 2 + x_dim;
    if(t_steps <= block_t) { 
        CalcBlockAndGridDim(block_t);
        dim3 dimBlock(block_dim_x, block_dim_y);
        dim3 dimGrid(grid_dim_x, grid_dim_y); 
        t_start = system_clock::now();
        GPUCPUStencilKernel<<<dimGrid, dimBlock>>>(cuda_stencil1, cuda_stencil2, x_dim, y_dim, z_dim, t_steps, alpha, beta_x_0, beta_x_1, 
            beta_y_0, beta_y_1, beta_z_0, beta_z_1);
        NaiveComputeBound(stencil.data(), stencil.data(), z_size, y_size, t_steps);
        cudaDeviceSynchronize();
        t_end = system_clock::now();
        t_duration = t_end - t_start;
        cout << "kernel time: " << t_duration.count() << endl;
        
        t_start = system_clock::now();
        CopyResultFromGPUToCPU(cuda_stencil2, stencil.data(), t_steps);
        cudaDeviceSynchronize();
        t_end = system_clock::now();
        t_duration = t_end - t_start;
        cout << "copy from gpu time: " << t_duration.count() << endl;
    }
    else { 
        int remain = t_steps % block_t;
        double *tmp = nullptr;
        for(int t = 0; t < t_steps - remain; t += block_t) { 
            GPUCPUStencilKernel<<<dimGrid, dimBlock>>>(cuda_stencil1, cuda_stencil2, x_dim, y_dim, z_dim, block_t, alpha, beta_x_0, beta_x_1,
                beta_y_0, beta_y_1, beta_z_0, beta_z_1); 
            NaiveComputeBound(stencil.data(), stencil.data(), z_size, y_size, block_t);
            cudaDeviceSynchronize(); 
            auto t_start = system_clock::now();
            CopyResultFromGPUToCPU(cuda_stencil2, stencil2.data(), block_t);
            CopyGPUResult(stencil2.data(), stencil.data(), block_t);
            auto t_end = system_clock::now();
            duration<double> t_duration = t_end - t_start;
            cout << "copy boundary from gpu time: " << t_duration.count() << endl; 
            t_start = system_clock::now();
            CopyBoundaryToGPU(stencil.data(), cuda_stencil2, y_size, z_size, block_t); 
            t_end = system_clock::now();
            t_duration = t_end - t_start;
            cout << "copy boundary to gpu time: " << t_duration.count() << endl;
            cudaDeviceSynchronize();
            tmp = cuda_stencil1;
            cuda_stencil1 = cuda_stencil2;
            cuda_stencil2 = tmp;
        }    
        if(remain > 0) { 
            CalcBlockAndGridDim(remain); 
            dim3 dimBlock(block_dim_x, block_dim_y);
            dim3 dimGrid(grid_dim_x, grid_dim_y);   
            GPUCPUStencilKernel<<<dimGrid, dimBlock>>>(cuda_stencil1, cuda_stencil2, x_dim, y_dim, z_dim, remain, alpha, beta_x_0, beta_x_1,
               beta_y_0, beta_y_1, beta_z_0, beta_z_1); 
            NaiveComputeBound(stencil.data(), stencil.data(), z_size, y_size, remain);
            auto t_start = system_clock::now();
            CopyResultFromGPUToCPU(cuda_stencil2, stencil2.data(), remain);
            CopyGPUResult(stencil2.data(), stencil.data(), remain);
            auto t_end = system_clock::now();
            duration<double> t_duration = t_end - t_start;
            cout << "copy boundary from gpu time: " << t_duration.count() << endl;
            cudaDeviceSynchronize(); 
        } 
        t_end = system_clock::now();
        t_duration = t_end - t_start;
        cout << "gpu kernel time: " << t_duration.count() << endl; 
    } 
     
    FreeGPUMem();
}
