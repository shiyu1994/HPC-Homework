//
//  matrix_multiplier.cpp
//  MatrixMultiply
//
//  Created by Shi Yu on 2017/12/17.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#include "matrix_multiplier.hpp"
#include <x86intrin.h>
#include <iostream>
#include "vector"
#include <mkl.h>
#include <omp.h>
#include <cassert>

using std::cout;
using std::endl;
using std::vector;

MatrixMultiplier::MatrixMultiplier(dmatrix64 &matrixx1, dmatrix64 &matrixx2,
                                   dmatrix64 &matrixx3, Config configg):
matrix1(matrixx1), matrix2(matrixx2), matrix3(matrixx3), config(configg) {
    
}

void MatrixMultiplier::Compute() {  
    ComputeStandAlone();
}

void MatrixMultiplier::ComputeStandAlone() {    
    /*int num_threads = config.num_threads;
    int block_size = config.block_size;
    
    int nblocks = padded_dim / block_size;
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < nblocks; ++i) {
        int row_start = i * block_size;
        int row_end = row_start + block_size;
        for(int j = 0; j < nblocks; ++j) {
            int col_start = j * block_size;
            int col_end = col_start + block_size;*/
    int num_threads = config.num_threads;                               
    int block_size = config.block_size;
            
    int nblocks = padded_dim / block_size;
    int blocks_per_thread = (nblocks + num_threads - 1) / num_threads;
    int remainder = nblocks - (nblocks / blocks_per_thread * blocks_per_thread);
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int block_id = 0; block_id < nblocks * nblocks; ++block_id) {
            int i = CalcRowBlockID(nblocks, nblocks, blocks_per_thread, remainder, block_id);
            int j = CalcColBlockID(nblocks, nblocks, blocks_per_thread, remainder, block_id);
            int col_start = block_size * j;
            int col_end = std::min(col_start + block_size, padded_dim);
            int row_start = block_size * i;
            int row_end = std::min(row_start + block_size, padded_dim);
            for(int k = 0; k < nblocks; ++k) {
                int k_start = block_size * k;
                int k_end = std::min(k_start + block_size, padded_dim); 
                int block_k_end = k_start + (k_end - k_start) / 4 * 4;
                for(int jj = col_start; jj < col_end; jj += 12) {
                    for(int ii = row_start; ii < row_end; ii += 4) {
                        __m256d output_row_0 = _mm256_load_pd(&matrix3[ii][jj]);
                        __m256d output_row_4 = _mm256_load_pd(&matrix3[ii][jj + 4]);
                        __m256d output_row_8 = _mm256_load_pd(&matrix3[ii][jj + 8]);
                        
                        __m256d output_row_1 = _mm256_load_pd(&matrix3[ii + 1][jj]);
                        __m256d output_row_5 = _mm256_load_pd(&matrix3[ii + 1][jj + 4]);        
                        __m256d output_row_9 = _mm256_load_pd(&matrix3[ii + 1][jj + 8]);
                        
                        __m256d output_row_2 = _mm256_load_pd(&matrix3[ii + 2][jj]);    
                        __m256d output_row_6 = _mm256_load_pd(&matrix3[ii + 2][jj + 4]);
                        __m256d output_row_10 = _mm256_load_pd(&matrix3[ii + 2][jj + 8]);
                        
                        __m256d output_row_3 = _mm256_load_pd(&matrix3[ii + 3][jj]);
                        __m256d output_row_7 = _mm256_load_pd(&matrix3[ii + 3][jj + 4]);
                        __m256d output_row_11 = _mm256_load_pd(&matrix3[ii + 3][jj + 8]);
                        __m256d a00;
                        
                        __m256d b0;
                        __m256d b4;
                        __m256d b8;
                        for(int kk = k_start; kk < block_k_end; kk += 4) {
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk]);
                            
                            b0 = _mm256_load_pd(&matrix2[kk][jj]);
                            b4 = _mm256_load_pd(&matrix2[kk][jj + 4]);
                            b8 = _mm256_load_pd(&matrix2[kk][jj + 8]);
                            
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                            
                            b0 = _mm256_load_pd(&matrix2[kk + 1][jj]);
                            b4 = _mm256_load_pd(&matrix2[kk + 1][jj + 4]);
                            b8 = _mm256_load_pd(&matrix2[kk + 1][jj + 8]);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk + 1]);
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 1]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 1]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 1]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                            
                            b0 = _mm256_load_pd(&matrix2[kk + 2][jj]);
                            b4 = _mm256_load_pd(&matrix2[kk + 2][jj + 4]);
                            b8 = _mm256_load_pd(&matrix2[kk + 2][jj + 8]);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk + 2]);
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 2]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 2]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 2]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                            
                            b0 = _mm256_load_pd(&matrix2[kk + 3][jj]);
                            b4 = _mm256_load_pd(&matrix2[kk + 3][jj + 4]);
                            b8 = _mm256_load_pd(&matrix2[kk + 3][jj + 8]);
                            
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk + 3]);
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 3]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 3]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 3]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                        }
                        _mm256_store_pd(&matrix3[ii][jj], output_row_0);
                        _mm256_store_pd(&matrix3[ii][jj + 4], output_row_4);
                        _mm256_store_pd(&matrix3[ii][jj + 8], output_row_8);
                        _mm256_store_pd(&matrix3[ii + 1][jj], output_row_1);
                        _mm256_store_pd(&matrix3[ii + 1][jj + 4], output_row_5);
                        _mm256_store_pd(&matrix3[ii + 1][jj + 8], output_row_9);
                        _mm256_store_pd(&matrix3[ii + 2][jj], output_row_2);
                        _mm256_store_pd(&matrix3[ii + 2][jj + 4], output_row_6);
                        _mm256_store_pd(&matrix3[ii + 2][jj + 8], output_row_10);
                        _mm256_store_pd(&matrix3[ii + 3][jj], output_row_3);
                        _mm256_store_pd(&matrix3[ii + 3][jj + 4], output_row_7);
                        _mm256_store_pd(&matrix3[ii + 3][jj + 8], output_row_11);
                    }
                }
            }
        }
    //}
}

void MatrixMultiplier::InitData() {
    int dim = config.dim;
    srand(static_cast<int>(std::time(nullptr)));       
    if(dim % config.block_size == 0) {
        padded_dim = dim;
    }
    else {
        padded_dim = dim + (config.block_size - dim % config.block_size);
    }
    matrix1.resize(padded_dim);
    matrix2.resize(padded_dim);
    matrix3.resize(padded_dim);
#pragma omp parallel for schedule(static) num_threads(config.num_threads)
    for(int i = 0; i < padded_dim; ++i) {
        matrix1[i].resize(padded_dim, 0.0);
        matrix2[i].resize(padded_dim, 0.0);
        matrix3[i].resize(padded_dim, 0.0);
        if(i < dim) {
            for(int j = 0; j < dim; ++j) {
                matrix1[i][j] = 1.0 * std::rand() / RAND_MAX;
                matrix2[i][j] = 1.0 * std::rand() / RAND_MAX;
            }
        }
    }
}

void MatrixMultiplier::Check() {
    /*int n = config.dim;
    double *A = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *B = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *C = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    
    double *AA = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *BB = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *CC = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    
    //prepare data for mkl
#pragma omp parallel for schedule(static) num_threads(config.num_threads)
     for(int i = 0; i < n; ++i) {
         for(int j = 0; j < n; ++j) {
         AA[i * n + j] = A[i * n + j] = matrix1[i][j];
         BB[i * n + j] = B[i * n + j] = matrix2[i][j];
         CC[i * n + j] = C[i * n + j] = 0.0;
         }
     }
     double time_start = omp_get_wtime();
     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 n, n, n, 1.0, A, n, B, n, 0.0, C, n);      
     double time_end = omp_get_wtime();
     cout << "mkl time: " << (time_end - time_start) << endl;
     
     for(int i = 0; i < n; ++i) {
         for(int j = 0; j < n; ++j) {
             if(!(C[i * n + j] - matrix3[i][j] <= 1e-10 && C[i * n + j] - matrix3[i][j] >= -1e-10)) {
                 cout << i << " " << j << endl;
                 cout << C[i * n + j] << " " << matrix3[i][j] << endl;
             }
             assert(C[i * n + j] - matrix3[i][j] <= 1e-10 && C[i * n + j] - matrix3[i][j] >= -1e-10);
         }
     }
    
    cout << "check pass" << endl;
    
     MKL_free(A);
     MKL_free(B);
     MKL_free(C);*/
}
