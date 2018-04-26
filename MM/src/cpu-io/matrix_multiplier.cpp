//
//  matrix_multiplier.cpp
//  MatrixMultiply
//
//  Created by Shi Yu on 2017/12/17.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#include "cpu_io_matrix_multiplier.hpp"
#include <x86intrin.h>
#include <iostream>
#include "vector"
//#include <mkl.h>
#include <omp.h>
#include <cassert>
#include <fstream>
#include <cstdio>
#include <ctime>

using std::cout;
using std::endl;
using std::vector;
using std::ifstream;
using std::ofstream;

MatrixMultiplier::MatrixMultiplier(string fname_A, string fname_B, string fname_C, Config configg):
fname_A(fname_A), fname_B(fname_B), fname_C(fname_C), config(configg) {}

void MatrixMultiplier::ReadLinesIntoMemory(string fname, int lines_begin,
                                           int lines_end, dmatrix64 &matrix,
                                           int padded_rows, int padded_cols) {
    ifstream fin(fname);
    double value;
    for(int i = 0; i < config.dim * lines_begin; ++i) {
        fin >> value;
    }
    
    matrix.clear();
    matrix.resize(padded_rows);
    
    for(int i = 0; i < matrix.size(); ++i) {
        matrix[i].resize(padded_cols, 0.0);
    }

    for(int i = 0; i < config.dim * (lines_end - lines_begin); ++i) {
        fin >> matrix[i / config.dim][i % config.dim];
    }
    
    fin.close();
}

void MatrixMultiplier::WriteOut(string fname, int rows, int cols, dmatrix64 &matrix) {
    ofstream fout(fname, ofstream::app | ofstream::out);
    fout.precision(10);     
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            fout << matrix[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

string MatrixMultiplier::Compute() {
    int stride = config.stride;
    int dim = config.dim;
    int padded_cols = config.dim;
    if(padded_cols % stride != 0) {
        padded_cols += (stride - padded_cols % stride);
    }
    string cur_fname_C = fname_C;
    for(int col = 0; col < dim; col += stride) {
        string next_fname_C = config.dir_name + "/C" + std::to_string(col);
        int line_b_begin = col;
        int line_b_end = std::min(col + stride, dim);
        int padded_lines_b = line_b_end - line_b_begin;
        if(padded_lines_b % stride != 0) {
            padded_lines_b += (stride - padded_lines_b % stride);
        }
        ReadLinesIntoMemory(fname_B, line_b_begin, line_b_end, matrix2, padded_lines_b, padded_cols);   
        for(int i = 0; i < dim; i += stride) {
            int line_a_begin = i;
            int line_a_end = std::min(i + stride, dim);
            int padded_lines_a = line_a_end - line_a_begin;
            if(padded_lines_a % stride != 0) {
                padded_lines_a += (stride - padded_lines_a % stride);
            }
            ReadLinesIntoMemory(fname_A, line_a_begin, line_a_end, matrix1, padded_lines_a, padded_cols);
            ReadLinesIntoMemory(cur_fname_C, line_a_begin, line_a_end, matrix3, padded_lines_a, padded_cols);   
            for(int k = 0; k < padded_cols; k += stride) {
                ComputeStandAlone(line_a_begin, col, line_b_begin, k);
            }
            WriteOut(next_fname_C, line_a_end - line_a_begin, config.dim, matrix3);
        }
        remove(cur_fname_C.c_str());        
        cur_fname_C = next_fname_C;
    }
    
    string out_fname = config.dir_name + "/C";
    ifstream fin(cur_fname_C);
    ofstream fout(out_fname);
    fout.precision(10);
    double value;
    for(int i = 0; i < config.dim; ++i) {
        for(int j = 0; j < config.dim; ++j) {
            fin >> value;
            fout << value << " ";
        }
        fout << endl; 
    }
    fin.close();
    fout.close();
    remove(cur_fname_C.c_str());                        
    
    return out_fname;
}

void MatrixMultiplier::ComputeStandAlone(int a_row_begin, int a_col_begin, int b_row_begin, int b_col_begin) {
    int num_threads = config.num_threads;
    int block_size = config.block_size;
    int nblocks = config.stride / block_size;
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < nblocks; ++i) {
        int row_start = i * block_size;
        int row_end = row_start + block_size;
        for(int j = 0; j < nblocks; ++j) {
            int col_start = j * block_size;
            int col_end = col_start + block_size;
            for(int k = 0; k < nblocks; ++k) {
                int k_start = block_size * k;
                int k_end = k_start + block_size;
                for(int jj = col_start; jj < col_end; jj += 12) {
                    for(int ii = row_start; ii < row_end; ii += 4) {
                        __m256d output_row_0 = _mm256_load_pd(&matrix3[ii][jj + b_col_begin]);
                        __m256d output_row_4 = _mm256_load_pd(&matrix3[ii][jj + 4 + b_col_begin]);
                        __m256d output_row_8 = _mm256_load_pd(&matrix3[ii][jj + 8 + b_col_begin]);
                        
                        __m256d output_row_1 = _mm256_load_pd(&matrix3[ii + 1][jj + b_col_begin]);
                        __m256d output_row_5 = _mm256_load_pd(&matrix3[ii + 1][jj + 4 + b_col_begin]);
                        __m256d output_row_9 = _mm256_load_pd(&matrix3[ii + 1][jj + 8 + b_col_begin]);
                        
                        __m256d output_row_2 = _mm256_load_pd(&matrix3[ii + 2][jj + b_col_begin]);
                        __m256d output_row_6 = _mm256_load_pd(&matrix3[ii + 2][jj + 4 + b_col_begin]);
                        __m256d output_row_10 = _mm256_load_pd(&matrix3[ii + 2][jj + 8 + b_col_begin]);
                        
                        __m256d output_row_3 = _mm256_load_pd(&matrix3[ii + 3][jj + b_col_begin]);
                        __m256d output_row_7 = _mm256_load_pd(&matrix3[ii + 3][jj + 4 + b_col_begin]);
                        __m256d output_row_11 = _mm256_load_pd(&matrix3[ii + 3][jj + 8 + b_col_begin]);
                        __m256d a00;
                        
                        __m256d b0;
                        __m256d b4;
                        __m256d b8;
                        for(int kk = k_start; kk < k_end; kk += 4) {
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk + a_col_begin]);
                            
                            b0 = _mm256_load_pd(&matrix2[kk][jj + b_col_begin]);
                            b4 = _mm256_load_pd(&matrix2[kk][jj + 4 + b_col_begin]);
                            b8 = _mm256_load_pd(&matrix2[kk][jj + 8 + b_col_begin]);
                            
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + a_col_begin]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + a_col_begin]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + a_col_begin]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                            
                            b0 = _mm256_load_pd(&matrix2[kk + 1][jj + b_col_begin]);
                            b4 = _mm256_load_pd(&matrix2[kk + 1][jj + 4 + b_col_begin]);
                            b8 = _mm256_load_pd(&matrix2[kk + 1][jj + 8 + b_col_begin]);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk + 1 + a_col_begin]);
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 1 + a_col_begin]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 1 + a_col_begin]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 1 + a_col_begin]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                            
                            b0 = _mm256_load_pd(&matrix2[kk + 2][jj + b_col_begin]);
                            b4 = _mm256_load_pd(&matrix2[kk + 2][jj + 4 + b_col_begin]);
                            b8 = _mm256_load_pd(&matrix2[kk + 2][jj + 8 + b_col_begin]);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk + 2 + a_col_begin]);
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 2 + a_col_begin]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 2 + a_col_begin]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 2 + a_col_begin]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                            
                            b0 = _mm256_load_pd(&matrix2[kk + 3][jj + b_col_begin]);
                            b4 = _mm256_load_pd(&matrix2[kk + 3][jj + 4 + b_col_begin]);
                            b8 = _mm256_load_pd(&matrix2[kk + 3][jj + 8 + b_col_begin]);
                            
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii][kk + 3 + a_col_begin]);
                            output_row_0 = _mm256_fmadd_pd(a00, b0, output_row_0);
                            output_row_4 = _mm256_fmadd_pd(a00, b4, output_row_4);
                            output_row_8 = _mm256_fmadd_pd(a00, b8, output_row_8);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 3 + a_col_begin]);
                            output_row_1 = _mm256_fmadd_pd(a00, b0, output_row_1);
                            output_row_5 = _mm256_fmadd_pd(a00, b4, output_row_5);
                            output_row_9 = _mm256_fmadd_pd(a00, b8, output_row_9);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 3 + a_col_begin]);
                            output_row_2 = _mm256_fmadd_pd(a00, b0, output_row_2);
                            output_row_6 = _mm256_fmadd_pd(a00, b4, output_row_6);
                            output_row_10 = _mm256_fmadd_pd(a00, b8, output_row_10);
                            
                            a00 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 3 + a_col_begin]);
                            output_row_3 = _mm256_fmadd_pd(a00, b0, output_row_3);
                            output_row_7 = _mm256_fmadd_pd(a00, b4, output_row_7);
                            output_row_11 = _mm256_fmadd_pd(a00, b8, output_row_11);
                        }
                        _mm256_store_pd(&matrix3[ii][jj + b_col_begin], output_row_0);
                        _mm256_store_pd(&matrix3[ii][jj + 4 + b_col_begin], output_row_4);
                        _mm256_store_pd(&matrix3[ii][jj + 8 + b_col_begin], output_row_8);
                        _mm256_store_pd(&matrix3[ii + 1][jj + b_col_begin], output_row_1);
                        _mm256_store_pd(&matrix3[ii + 1][jj + 4 + b_col_begin], output_row_5);
                        _mm256_store_pd(&matrix3[ii + 1][jj + 8 + b_col_begin], output_row_9);
                        _mm256_store_pd(&matrix3[ii + 2][jj + b_col_begin], output_row_2);
                        _mm256_store_pd(&matrix3[ii + 2][jj + 4 + b_col_begin], output_row_6);
                        _mm256_store_pd(&matrix3[ii + 2][jj + 8 + b_col_begin], output_row_10);
                        _mm256_store_pd(&matrix3[ii + 3][jj + b_col_begin], output_row_3);
                        _mm256_store_pd(&matrix3[ii + 3][jj + 4 + b_col_begin], output_row_7);
                        _mm256_store_pd(&matrix3[ii + 3][jj + 8 + b_col_begin], output_row_11);
                    }
                }
            }
        }
    }
}

void MatrixMultiplier::StandAloneInitData() {
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
    
    //prepare data for mkl
#pragma omp parallel for schedule(static) num_threads(config.num_threads)
     for(int i = 0; i < n; ++i) {
         for(int j = 0; j < n; ++j) {
             A[i * n + j] = matrix1[i][j];
             B[i * n + j] = matrix2[i][j];
             C[i * n + j] = 0.0;
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
