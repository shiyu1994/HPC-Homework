//
//  gpu_matrix_multiplier.cpp
//  GPUMatrixMultiplier
//
//  Created by Shi Yu on 2017/12/23.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#include "gpu_matrix_multiplier.hpp"
#include "cuda.h"
#include <iostream>
#include <ctime>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::system_clock;
using std::chrono::duration;

//use 32 x 32 threads per block
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 8 
#define SHARE_MEM_X 64 
#define SHARE_MEM_K 16
#define SHARE_MEM_Y 64 

GPUMatrixMultiplier::GPUMatrixMultiplier(const dvec64 &matrixx1, const dvec64 &matrixx2, dvec64 &matrixx3, int dimm):
matrix1(matrixx1), matrix2(matrixx2), matrix3(matrixx3), dim(dimm) {}

void GPUMatrixMultiplier::AllocGPUMem() {
    //allocate memory, padding the threadBlocks 
    dim_pad = (dim + SHARE_MEM_X - 1) / SHARE_MEM_X * SHARE_MEM_X;
    
    size_t cuda_matrix_size = sizeof(double) * dim_pad * dim_pad;
    cudaMalloc(&cuda_matrix1, cuda_matrix_size);
    cudaMalloc(&cuda_matrix2, cuda_matrix_size);
    cudaMalloc(&cuda_matrix3, cuda_matrix_size);
    
    cudaMemset(cuda_matrix1, 0, cuda_matrix_size);
    cudaMemset(cuda_matrix2, 0, cuda_matrix_size);
}

void GPUMatrixMultiplier::CopyDataFromCPUToGPU() {
    //PrintMatrix(matrix1, dim);
    //PrintMatrix(matrix2, dim);
    size_t row_size = sizeof(double) * dim; 
    double *cuda_matrix1_ptr = cuda_matrix1, *cuda_matrix2_ptr = cuda_matrix2;
    const double *matrix1_ptr = matrix1.data(), *matrix2_ptr = matrix2.data();
    for(int i = 0; i < dim; ++i) {
        cudaMemcpy(cuda_matrix1_ptr, matrix1_ptr, row_size, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_matrix2_ptr, matrix2_ptr, row_size, cudaMemcpyHostToDevice);
        cuda_matrix1_ptr += dim_pad;
        cuda_matrix2_ptr += dim_pad;
        matrix1_ptr += dim;
        matrix2_ptr += dim;
    }
}

void GPUMatrixMultiplier::FreeGPUMem() {
    cudaFree(cuda_matrix1);
    cudaFree(cuda_matrix2);
    cudaFree(cuda_matrix3);
}

void GPUMatrixMultiplier::CopyResultFromGPUToCPU() {
    size_t row_size = sizeof(double) * dim; 
    double *cuda_matrix3_ptr = cuda_matrix3;
    double *matrix3_ptr = matrix3.data();
    for(int i = 0; i < dim; ++i) {
        cudaMemcpy(matrix3_ptr, cuda_matrix3_ptr, row_size, cudaMemcpyDeviceToHost);
        cuda_matrix3_ptr += dim_pad;
        matrix3_ptr += dim;
    }
}

__global__ void GPUMatrixMultiplyKernel(double const* __restrict__ cuda_matrix1, double const* __restrict__ cuda_matrix2, double *cuda_matrix3, int dim_pad) { 
    __shared__ double mat1[SHARE_MEM_K][SHARE_MEM_Y + 1], mat2[SHARE_MEM_K][SHARE_MEM_X + 1];
 
    const int row = blockIdx.y * SHARE_MEM_Y + threadIdx.y;
    const int col = blockIdx.x * SHARE_MEM_X + threadIdx.x;
    
    const int index = row * dim_pad + col;
    double sum[8][4], a[8], b[4];
     
    sum[0][0] = cuda_matrix3[index];
    sum[0][1] = cuda_matrix3[index + 16];
    sum[0][2] = cuda_matrix3[index + 32];
    sum[0][3] = cuda_matrix3[index + 48];
    
    sum[1][0] = cuda_matrix3[index + dim_pad * 8];
    sum[1][1] = cuda_matrix3[index + dim_pad * 8 + 16];
    sum[1][2] = cuda_matrix3[index + dim_pad * 8 + 32];
    sum[1][3] = cuda_matrix3[index + dim_pad * 8 + 48]; 
    
    sum[2][0] = cuda_matrix3[index + dim_pad * 16];
    sum[2][1] = cuda_matrix3[index + dim_pad * 16 + 16];
    sum[2][2] = cuda_matrix3[index + dim_pad * 16 + 32];
    sum[2][3] = cuda_matrix3[index + dim_pad * 16 + 48];
    
    sum[3][0] = cuda_matrix3[index + dim_pad * 24];
    sum[3][1] = cuda_matrix3[index + dim_pad * 24 + 16];
    sum[3][2] = cuda_matrix3[index + dim_pad * 24 + 32];
    sum[3][3] = cuda_matrix3[index + dim_pad * 24 + 48];
    
    
    sum[4][0] = cuda_matrix3[index + 32 * dim_pad];
    sum[4][1] = cuda_matrix3[index + 16 + 32 * dim_pad];
    sum[4][2] = cuda_matrix3[index + 32 + 32 * dim_pad];
    sum[4][3] = cuda_matrix3[index + 48 + 32 * dim_pad];
   
    
    sum[5][0] = cuda_matrix3[index + dim_pad * 40];
    sum[5][1] = cuda_matrix3[index + dim_pad * 40 + 16];
    sum[5][2] = cuda_matrix3[index + dim_pad * 40 + 32];
    sum[5][3] = cuda_matrix3[index + dim_pad * 40 + 48]; 
    

    sum[6][0] = cuda_matrix3[index + dim_pad * 48];
    sum[6][1] = cuda_matrix3[index + dim_pad * 48 + 16];
    sum[6][2] = cuda_matrix3[index + dim_pad * 48 + 32];
    sum[6][3] = cuda_matrix3[index + dim_pad * 48 + 48];
    
    
    sum[7][0] = cuda_matrix3[index + dim_pad * 56];
    sum[7][1] = cuda_matrix3[index + dim_pad * 56 + 16];
    sum[7][2] = cuda_matrix3[index + dim_pad * 56 + 32];
    sum[7][3] = cuda_matrix3[index + dim_pad * 56 + 48];
    

    for(int i = 0; i < dim_pad; i += SHARE_MEM_K) { 
        //each thread reads 8 element into shared memory
        const int mat1_index = (blockIdx.y * SHARE_MEM_Y + threadIdx.y) * dim_pad + i + threadIdx.x;
        const int mat2_index = (i + threadIdx.y) * dim_pad + col; 
        mat1[threadIdx.x][threadIdx.y] = cuda_matrix1[mat1_index]; 
        mat1[threadIdx.x][threadIdx.y + 8] = cuda_matrix1[mat1_index + 8 * dim_pad]; 
        mat1[threadIdx.x][threadIdx.y + 16] = cuda_matrix1[mat1_index + 16 * dim_pad]; 
        mat1[threadIdx.x][threadIdx.y + 24] = cuda_matrix1[mat1_index + 24 * dim_pad]; 
        mat1[threadIdx.x][threadIdx.y + 32] = cuda_matrix1[mat1_index + 32 * dim_pad]; 
        mat1[threadIdx.x][threadIdx.y + 40] = cuda_matrix1[mat1_index + 40 * dim_pad]; 
        mat1[threadIdx.x][threadIdx.y + 48] = cuda_matrix1[mat1_index + 48 * dim_pad]; 
        mat1[threadIdx.x][threadIdx.y + 56] = cuda_matrix1[mat1_index + 56 * dim_pad]; 
        

        mat2[threadIdx.y][threadIdx.x] = cuda_matrix2[mat2_index];  
        mat2[threadIdx.y][threadIdx.x + 16] = cuda_matrix2[mat2_index + 16];  
        mat2[threadIdx.y][threadIdx.x + 32] = cuda_matrix2[mat2_index + 32]; 
        mat2[threadIdx.y][threadIdx.x + 48] = cuda_matrix2[mat2_index + 48];    
        mat2[threadIdx.y + 8][threadIdx.x] = cuda_matrix2[mat2_index + 8 * dim_pad];  
        mat2[threadIdx.y + 8][threadIdx.x + 16] = cuda_matrix2[mat2_index + 16 + 8 * dim_pad];  
        mat2[threadIdx.y + 8][threadIdx.x + 32] = cuda_matrix2[mat2_index + 32 + 8 * dim_pad]; 
        mat2[threadIdx.y + 8][threadIdx.x + 48] = cuda_matrix2[mat2_index + 48 + 8 * dim_pad];    
        
        __syncthreads();
        //each thread calculates 8 element
#pragma unroll(16)
        for(int j = 0; j < 16; ++j) {
            a[0] = mat1[j][threadIdx.y];
            a[1] = mat1[j][threadIdx.y + 8];
            a[2] = mat1[j][threadIdx.y + 16];
            a[3] = mat1[j][threadIdx.y + 24];
            a[4] = mat1[j][threadIdx.y + 32];
            a[5] = mat1[j][threadIdx.y + 40];
            a[6] = mat1[j][threadIdx.y + 48];
            a[7] = mat1[j][threadIdx.y + 56];


            b[0] = mat2[j][threadIdx.x];
            b[1] = mat2[j][threadIdx.x + 16];
            b[2] = mat2[j][threadIdx.x + 32];
            b[3] = mat2[j][threadIdx.x + 48];
            
            sum[0][0] += a[0] * b[0];
            sum[0][1] += a[0] * b[1];
            sum[1][0] += a[1] * b[0];
            sum[1][1] += a[1] * b[1];
            
            sum[0][2] += a[0] * b[2];
            sum[0][3] += a[0] * b[3];
            sum[1][2] += a[1] * b[2];
            sum[1][3] += a[1] * b[3];
            
            sum[2][0] += a[2] * b[0];
            sum[2][1] += a[2] * b[1];
            sum[3][0] += a[3] * b[0];
            sum[3][1] += a[3] * b[1]; 

            sum[2][2] += a[2] * b[2];
            sum[2][3] += a[2] * b[3];
            sum[3][2] += a[3] * b[2];
            sum[3][3] += a[3] * b[3]; 

            sum[4][0] += a[4] * b[0];
            sum[4][1] += a[4] * b[1];
            sum[5][0] += a[5] * b[0];
            sum[5][1] += a[5] * b[1];
            
            sum[4][2] += a[4] * b[2];
            sum[4][3] += a[4] * b[3];
            sum[5][2] += a[5] * b[2];
            sum[5][3] += a[5] * b[3];
            
            sum[6][0] += a[6] * b[0];
            sum[6][1] += a[6] * b[1];
            sum[7][0] += a[7] * b[0];
            sum[7][1] += a[7] * b[1]; 

            sum[6][2] += a[6] * b[2];
            sum[6][3] += a[6] * b[3];
            sum[7][2] += a[7] * b[2];
            sum[7][3] += a[7] * b[3];             
        }
        __syncthreads();
    }
    cuda_matrix3[index] =sum[0][0] ;
    cuda_matrix3[index + 16] =sum[0][1] ;
    cuda_matrix3[index + 32] =sum[0][2] ;
    cuda_matrix3[index + 48] =sum[0][3] ;
    
    cuda_matrix3[index + dim_pad * 8] =sum[1][0] ;
    cuda_matrix3[index + dim_pad * 8 + 16] =sum[1][1] ;
    cuda_matrix3[index + dim_pad * 8 + 32] =sum[1][2] ;
    cuda_matrix3[index + dim_pad * 8 + 48] =sum[1][3]  ;
    
    cuda_matrix3[index + dim_pad * 16] = sum[2][0]  ;
    cuda_matrix3[index + dim_pad * 16 + 16] = sum[2][1]  ;
    cuda_matrix3[index + dim_pad * 16 + 32]=sum[2][2]  ;
    cuda_matrix3[index + dim_pad * 16 + 48]=sum[2][3]  ;
    
    cuda_matrix3[index + dim_pad * 24]=sum[3][0]  ;
    cuda_matrix3[index + dim_pad * 24 + 16]=sum[3][1]  ;
    cuda_matrix3[index + dim_pad * 24 + 32]=sum[3][2]  ;
    cuda_matrix3[index + dim_pad * 24 + 48]=sum[3][3]  ;
    
    
    cuda_matrix3[index + 32 * dim_pad]=sum[4][0]  ;
    cuda_matrix3[index + 16 + 32 * dim_pad]=sum[4][1]  ;
    cuda_matrix3[index + 32 + 32 * dim_pad]=sum[4][2]  ;
    cuda_matrix3[index + 48 + 32 * dim_pad]=sum[4][3]  ;
    
    
    cuda_matrix3[index + dim_pad * 40]=sum[5][0]  ;
    cuda_matrix3[index + dim_pad * 40 + 16]=sum[5][1]  ;
    cuda_matrix3[index + dim_pad * 40 + 32]=sum[5][2]  ;
    cuda_matrix3[index + dim_pad * 40 + 48]=sum[5][3]  ; 
    

    cuda_matrix3[index + dim_pad * 48]=sum[6][0]  ;
    cuda_matrix3[index + dim_pad * 48 + 16]=sum[6][1]  ;
    cuda_matrix3[index + dim_pad * 48 + 32]=sum[6][2]  ;
    cuda_matrix3[index + dim_pad * 48 + 48]=sum[6][3]  ;
       
    cuda_matrix3[index + dim_pad * 56]=sum[7][0]  ;
    cuda_matrix3[index + dim_pad * 56 + 16]=sum[7][1]  ;
    cuda_matrix3[index + dim_pad * 56 + 32]=sum[7][2]  ;
    cuda_matrix3[index + dim_pad * 56 + 48]=sum[7][3]  ;
    }

void GPUMatrixMultiplier::Compute() {
    cout << "allocating gpu memory" << endl;
    auto t_start = system_clock::now(); 
    AllocGPUMem();
    auto t_end = system_clock::now(); 
    duration<double> t_duration = t_end - t_start;
    cout << "allocate gpu memory time: " << t_duration.count() << endl;

    cout << "copying data to gpu" << endl;
    t_start = system_clock::now();
    CopyDataFromCPUToGPU();
    t_end = system_clock::now();
    t_duration = t_end - t_start;
    cout << "copy data to gpu time: " << t_duration.count() << endl;

    int grid_dim_x = 0, grid_dim_y = 0, block_dim_x = 0, block_dim_y = 0;
    CalcGridAndBlockDim(grid_dim_x, grid_dim_y, block_dim_x, block_dim_y);
    dim3 dimBlock(block_dim_x, block_dim_y);
    dim3 dimGrid(grid_dim_x, grid_dim_y);
    cout << "kernel start" << endl;
    t_start = system_clock::now(); 
    GPUMatrixMultiplyKernel<<<dimGrid, dimBlock>>>(cuda_matrix1, cuda_matrix2, cuda_matrix3, dim_pad);
    
    //synchronize before record time
    cudaDeviceSynchronize();
    t_end = system_clock::now();
    t_duration = t_end - t_start; 
    cout << "kernel end" << endl;
    cout << "kernel time: " << t_duration.count() << endl;

    cout << "copying result from gpu" << endl;
    t_start = system_clock::now(); 
    CopyResultFromGPUToCPU(); 
    t_end = system_clock::now();
    t_duration = t_end - t_start;
    cout << "copy result from gpu time: " << t_duration.count() << endl;
    //PrintMatrix(matrix3, dim);

    FreeGPUMem();
}

void GPUMatrixMultiplier::CalcGridAndBlockDim(int &grid_dim_x, int &grid_dim_y, int &block_dim_x, int &block_dim_y) {
    //use 32 x 4 threads per block
    block_dim_x = BLOCK_DIM_X;
    block_dim_y = BLOCK_DIM_Y;
    //calculates block dim
    grid_dim_x = (dim + SHARE_MEM_X - 1) / SHARE_MEM_X;
    grid_dim_y = (dim + SHARE_MEM_Y - 1) / SHARE_MEM_Y; 
}

void GPUMatrixMultiplier::PrintMatrix(const dvec64 &matrix, int dim) {
    for(int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            cout << "(" << i << "," << j << ")" << matrix[i * dim + j] << " "; 
        }
        cout << endl;
    }
    cout << endl; 
}
