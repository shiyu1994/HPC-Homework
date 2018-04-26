#include <omp.h>
#include "knl_matrix_multiplier.hpp"
#include <vector>
#include <cstdlib>
#include <x86intrin.h>

using std::vector;
using std::endl;
using std::cout;

/*KNLMatrixMultiplier::KNLMatrixMultiplier(double **matrixx1, double **matrixx2, double **matrixx3, int dimm, int num_threadss, int block_sizee):
        matrix1(matrixx1), matrix2(matrixx2), matrix3(matrixx3) {
    dim = dimm;
    num_threads = num_threadss;
    block_size = block_sizee;
    if(dim % block_size == 0) {
        padded_dim = dim;
    }
    else {
        padded_dim = dim + (block_size - dim % block_size);
    }
}*/

KNLMatrixMultiplier::KNLMatrixMultiplier(dmatrix64 &matrixx1, dmatrix64 &matrixx2, dmatrix64 &matrixx3, int dimm, int num_threadss, int block_sizee):
        matrix1(matrixx1), matrix2(matrixx2), matrix3(matrixx3) {
    dim = dimm;
    num_threads = num_threadss;
    block_size = block_sizee;
    if(dim % block_size == 0) {
        padded_dim = dim;
    }
    else {
        padded_dim = dim + (block_size - dim % block_size);
    }
}

/*void KNLMatrixMultiplier::InitData() {
    srand(static_cast<int>(time(nullptr)));
    matrix1 = new double*[dim];
    matrix2 = new double*[dim];
    matrix3 = new double*[dim];
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < dim; ++i) {
        matrix1[i] = new double[dim];
        matrix2[i]  = new double[dim];
        matrix3[i]  = new double[dim];
        for(int j = 0; j < dim; ++j) {
            matrix1[i][j] = std::rand() * 1.0 / RAND_MAX;
            matrix2[i][j] = std::rand() * 1.0 / RAND_MAX;
        }
    } 
}*/

void KNLMatrixMultiplier::InitData() {
    srand(static_cast<int>(time(nullptr)));
    matrix1.resize(padded_dim);
    matrix2.resize(padded_dim);
    matrix3.resize(padded_dim);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < padded_dim; ++i) {
        matrix1[i].resize(padded_dim, 0.0);
        matrix2[i].resize(padded_dim, 0.0);
        matrix3[i].resize(padded_dim, 0.0);
        if(i < dim) {
        for(int j = 0; j < dim; ++j) {
            matrix1[i][j] = std::rand() * 1.0 / RAND_MAX;
            matrix2[i][j] = std::rand() * 1.0 / RAND_MAX;
        }
        }
    } 
}


void KNLMatrixMultiplier::Compute() {  
    //L2 blocking for 1M L2 cache, 192 * 192 * 3 * 8 = 884736 bytes
    //int block_size = 192;
    
    int nblocks = (padded_dim + block_size - 1) / block_size;
   
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
        int block_row_end = row_start + (row_end - row_start) / 4 * 4;
        int block_col_end = col_start + (col_end - col_start) / 48 * 48;
        for(int k = 0; k < nblocks; ++k) {
            int k_start = block_size * k;
            int k_end = std::min(k_start + block_size, padded_dim);
            int block_k_end = k_start + (k_end - k_start) / 4 * 4;
            for(int jj = col_start; jj < block_col_end; jj += 48) {
                for(int ii = row_start; ii < block_row_end; ii += 4) {
                    //use 31 512-bit registers, 24 for output matrix, 1 for matrix1, 6 for matrix2
                    __m512d output_row_0_0 = _mm512_load_pd(&matrix3[ii][jj]);
                    __m512d output_row_0_1 = _mm512_load_pd(&matrix3[ii][jj + 8]);
                    __m512d output_row_0_2 = _mm512_load_pd(&matrix3[ii][jj + 16]);
                    __m512d output_row_0_3 = _mm512_load_pd(&matrix3[ii][jj + 24]);
                    __m512d output_row_0_4 = _mm512_load_pd(&matrix3[ii][jj + 32]);
                    __m512d output_row_0_5 = _mm512_load_pd(&matrix3[ii][jj + 40]);

                    __m512d output_row_1_0 = _mm512_load_pd(&matrix3[ii + 1][jj]);
                    __m512d output_row_1_1 = _mm512_load_pd(&matrix3[ii + 1][jj + 8]);
                    __m512d output_row_1_2 = _mm512_load_pd(&matrix3[ii + 1][jj + 16]);
                    __m512d output_row_1_3 = _mm512_load_pd(&matrix3[ii + 1][jj + 24]);
                    __m512d output_row_1_4 = _mm512_load_pd(&matrix3[ii + 1][jj + 32]);
                    __m512d output_row_1_5 = _mm512_load_pd(&matrix3[ii + 1][jj + 40]);


                    __m512d output_row_2_0 = _mm512_load_pd(&matrix3[ii + 2][jj]);
                    __m512d output_row_2_1 = _mm512_load_pd(&matrix3[ii + 2][jj + 8]);
                    __m512d output_row_2_2 = _mm512_load_pd(&matrix3[ii + 2][jj + 16]);
                    __m512d output_row_2_3 = _mm512_load_pd(&matrix3[ii + 2][jj + 24]);
                    __m512d output_row_2_4 = _mm512_load_pd(&matrix3[ii + 2][jj + 32]);
                    __m512d output_row_2_5 = _mm512_load_pd(&matrix3[ii + 2][jj + 40]);

                    __m512d output_row_3_0 = _mm512_load_pd(&matrix3[ii + 3][jj]);
                    __m512d output_row_3_1 = _mm512_load_pd(&matrix3[ii + 3][jj + 8]);
                    __m512d output_row_3_2 = _mm512_load_pd(&matrix3[ii + 3][jj + 16]);
                    __m512d output_row_3_3 = _mm512_load_pd(&matrix3[ii + 3][jj + 24]);
                    __m512d output_row_3_4 = _mm512_load_pd(&matrix3[ii + 3][jj + 32]);
                    __m512d output_row_3_5 = _mm512_load_pd(&matrix3[ii + 3][jj + 40]);


                    __m256d a0;
                    __m512d a00;
                    
                    __m512d b0;
                    __m512d b1;
                    __m512d b2;
                    __m512d b3;
                    __m512d b4;
                    __m512d b5;
                    for(int kk = k_start; kk < block_k_end; kk += 4) {
                        
                        b0 = _mm512_load_pd(&matrix2[kk][jj]);
                        b1 = _mm512_load_pd(&matrix2[kk][jj + 8]);
                        b2 = _mm512_load_pd(&matrix2[kk][jj + 16]);
                        b3 = _mm512_load_pd(&matrix2[kk][jj + 24]);
                        b4 = _mm512_load_pd(&matrix2[kk][jj + 32]);
                        b5 = _mm512_load_pd(&matrix2[kk][jj + 40]);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii][kk]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_0_0 = _mm512_fmadd_pd(a00, b0, output_row_0_0);
                        output_row_0_1 = _mm512_fmadd_pd(a00, b1, output_row_0_1);
                        output_row_0_2 = _mm512_fmadd_pd(a00, b2, output_row_0_2);
                        output_row_0_3 = _mm512_fmadd_pd(a00, b3, output_row_0_3);
                        output_row_0_4 = _mm512_fmadd_pd(a00, b4, output_row_0_4);
                        output_row_0_5 = _mm512_fmadd_pd(a00, b5, output_row_0_5);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii + 1][kk]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_1_0 = _mm512_fmadd_pd(a00, b0, output_row_1_0);
                        output_row_1_1 = _mm512_fmadd_pd(a00, b1, output_row_1_1);
                        output_row_1_2 = _mm512_fmadd_pd(a00, b2, output_row_1_2);
                        output_row_1_3 = _mm512_fmadd_pd(a00, b3, output_row_1_3);
                        output_row_1_4 = _mm512_fmadd_pd(a00, b4, output_row_1_4);
                        output_row_1_5 = _mm512_fmadd_pd(a00, b5, output_row_1_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 2][kk]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_2_0 = _mm512_fmadd_pd(a00, b0, output_row_2_0);
                        output_row_2_1 = _mm512_fmadd_pd(a00, b1, output_row_2_1);
                        output_row_2_2 = _mm512_fmadd_pd(a00, b2, output_row_2_2);
                        output_row_2_3 = _mm512_fmadd_pd(a00, b3, output_row_2_3);
                        output_row_2_4 = _mm512_fmadd_pd(a00, b4, output_row_2_4);
                        output_row_2_5 = _mm512_fmadd_pd(a00, b5, output_row_2_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 3][kk]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_3_0 = _mm512_fmadd_pd(a00, b0, output_row_3_0);
                        output_row_3_1 = _mm512_fmadd_pd(a00, b1, output_row_3_1);
                        output_row_3_2 = _mm512_fmadd_pd(a00, b2, output_row_3_2);
                        output_row_3_3 = _mm512_fmadd_pd(a00, b3, output_row_3_3);
                        output_row_3_4 = _mm512_fmadd_pd(a00, b4, output_row_3_4);
                        output_row_3_5 = _mm512_fmadd_pd(a00, b5, output_row_3_5);


                        b0 = _mm512_load_pd(&matrix2[kk + 1][jj]);
                        b1 = _mm512_load_pd(&matrix2[kk + 1][jj + 8]);
                        b2 = _mm512_load_pd(&matrix2[kk + 1][jj + 16]);
                        b3 = _mm512_load_pd(&matrix2[kk + 1][jj + 24]);
                        b4 = _mm512_load_pd(&matrix2[kk + 1][jj + 32]);
                        b5 = _mm512_load_pd(&matrix2[kk + 1][jj + 40]);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii][kk + 1]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_0_0 = _mm512_fmadd_pd(a00, b0, output_row_0_0);
                        output_row_0_1 = _mm512_fmadd_pd(a00, b1, output_row_0_1);
                        output_row_0_2 = _mm512_fmadd_pd(a00, b2, output_row_0_2);
                        output_row_0_3 = _mm512_fmadd_pd(a00, b3, output_row_0_3);
                        output_row_0_4 = _mm512_fmadd_pd(a00, b4, output_row_0_4);
                        output_row_0_5 = _mm512_fmadd_pd(a00, b5, output_row_0_5);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 1]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_1_0 = _mm512_fmadd_pd(a00, b0, output_row_1_0);
                        output_row_1_1 = _mm512_fmadd_pd(a00, b1, output_row_1_1);
                        output_row_1_2 = _mm512_fmadd_pd(a00, b2, output_row_1_2);
                        output_row_1_3 = _mm512_fmadd_pd(a00, b3, output_row_1_3);
                        output_row_1_4 = _mm512_fmadd_pd(a00, b4, output_row_1_4);
                        output_row_1_5 = _mm512_fmadd_pd(a00, b5, output_row_1_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 1]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_2_0 = _mm512_fmadd_pd(a00, b0, output_row_2_0);
                        output_row_2_1 = _mm512_fmadd_pd(a00, b1, output_row_2_1);
                        output_row_2_2 = _mm512_fmadd_pd(a00, b2, output_row_2_2);
                        output_row_2_3 = _mm512_fmadd_pd(a00, b3, output_row_2_3);
                        output_row_2_4 = _mm512_fmadd_pd(a00, b4, output_row_2_4);
                        output_row_2_5 = _mm512_fmadd_pd(a00, b5, output_row_2_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 1]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_3_0 = _mm512_fmadd_pd(a00, b0, output_row_3_0);
                        output_row_3_1 = _mm512_fmadd_pd(a00, b1, output_row_3_1);
                        output_row_3_2 = _mm512_fmadd_pd(a00, b2, output_row_3_2);
                        output_row_3_3 = _mm512_fmadd_pd(a00, b3, output_row_3_3);
                        output_row_3_4 = _mm512_fmadd_pd(a00, b4, output_row_3_4);
                        output_row_3_5 = _mm512_fmadd_pd(a00, b5, output_row_3_5);

                        
                        b0 = _mm512_load_pd(&matrix2[kk + 2][jj]);
                        b1 = _mm512_load_pd(&matrix2[kk + 2][jj + 8]);
                        b2 = _mm512_load_pd(&matrix2[kk + 2][jj + 16]);
                        b3 = _mm512_load_pd(&matrix2[kk + 2][jj + 24]);
                        b4 = _mm512_load_pd(&matrix2[kk + 2][jj + 32]);
                        b5 = _mm512_load_pd(&matrix2[kk + 2][jj + 40]);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii][kk + 2]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_0_0 = _mm512_fmadd_pd(a00, b0, output_row_0_0);
                        output_row_0_1 = _mm512_fmadd_pd(a00, b1, output_row_0_1);
                        output_row_0_2 = _mm512_fmadd_pd(a00, b2, output_row_0_2);
                        output_row_0_3 = _mm512_fmadd_pd(a00, b3, output_row_0_3);
                        output_row_0_4 = _mm512_fmadd_pd(a00, b4, output_row_0_4);
                        output_row_0_5 = _mm512_fmadd_pd(a00, b5, output_row_0_5);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 2]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_1_0 = _mm512_fmadd_pd(a00, b0, output_row_1_0);
                        output_row_1_1 = _mm512_fmadd_pd(a00, b1, output_row_1_1);
                        output_row_1_2 = _mm512_fmadd_pd(a00, b2, output_row_1_2);
                        output_row_1_3 = _mm512_fmadd_pd(a00, b3, output_row_1_3);
                        output_row_1_4 = _mm512_fmadd_pd(a00, b4, output_row_1_4);
                        output_row_1_5 = _mm512_fmadd_pd(a00, b5, output_row_1_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 2]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_2_0 = _mm512_fmadd_pd(a00, b0, output_row_2_0);
                        output_row_2_1 = _mm512_fmadd_pd(a00, b1, output_row_2_1);
                        output_row_2_2 = _mm512_fmadd_pd(a00, b2, output_row_2_2);
                        output_row_2_3 = _mm512_fmadd_pd(a00, b3, output_row_2_3);
                        output_row_2_4 = _mm512_fmadd_pd(a00, b4, output_row_2_4);
                        output_row_2_5 = _mm512_fmadd_pd(a00, b5, output_row_2_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 2]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_3_0 = _mm512_fmadd_pd(a00, b0, output_row_3_0);
                        output_row_3_1 = _mm512_fmadd_pd(a00, b1, output_row_3_1);
                        output_row_3_2 = _mm512_fmadd_pd(a00, b2, output_row_3_2);
                        output_row_3_3 = _mm512_fmadd_pd(a00, b3, output_row_3_3);
                        output_row_3_4 = _mm512_fmadd_pd(a00, b4, output_row_3_4);
                        output_row_3_5 = _mm512_fmadd_pd(a00, b5, output_row_3_5);


                        b0 = _mm512_load_pd(&matrix2[kk + 3][jj]);
                        b1 = _mm512_load_pd(&matrix2[kk + 3][jj + 8]);
                        b2 = _mm512_load_pd(&matrix2[kk + 3][jj + 16]);
                        b3 = _mm512_load_pd(&matrix2[kk + 3][jj + 24]);
                        b4 = _mm512_load_pd(&matrix2[kk + 3][jj + 32]);
                        b5 = _mm512_load_pd(&matrix2[kk + 3][jj + 40]);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii][kk + 3]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_0_0 = _mm512_fmadd_pd(a00, b0, output_row_0_0);
                        output_row_0_1 = _mm512_fmadd_pd(a00, b1, output_row_0_1);
                        output_row_0_2 = _mm512_fmadd_pd(a00, b2, output_row_0_2);
                        output_row_0_3 = _mm512_fmadd_pd(a00, b3, output_row_0_3);
                        output_row_0_4 = _mm512_fmadd_pd(a00, b4, output_row_0_4);
                        output_row_0_5 = _mm512_fmadd_pd(a00, b5, output_row_0_5);
                        
                        a0 = _mm256_broadcast_sd(&matrix1[ii + 1][kk + 3]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_1_0 = _mm512_fmadd_pd(a00, b0, output_row_1_0);
                        output_row_1_1 = _mm512_fmadd_pd(a00, b1, output_row_1_1);
                        output_row_1_2 = _mm512_fmadd_pd(a00, b2, output_row_1_2);
                        output_row_1_3 = _mm512_fmadd_pd(a00, b3, output_row_1_3);
                        output_row_1_4 = _mm512_fmadd_pd(a00, b4, output_row_1_4);
                        output_row_1_5 = _mm512_fmadd_pd(a00, b5, output_row_1_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 2][kk + 3]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_2_0 = _mm512_fmadd_pd(a00, b0, output_row_2_0);
                        output_row_2_1 = _mm512_fmadd_pd(a00, b1, output_row_2_1);
                        output_row_2_2 = _mm512_fmadd_pd(a00, b2, output_row_2_2);
                        output_row_2_3 = _mm512_fmadd_pd(a00, b3, output_row_2_3);
                        output_row_2_4 = _mm512_fmadd_pd(a00, b4, output_row_2_4);
                        output_row_2_5 = _mm512_fmadd_pd(a00, b5, output_row_2_5);

                        a0 = _mm256_broadcast_sd(&matrix1[ii + 3][kk + 3]);
                        a00 = _mm512_broadcast_f64x4(a0);
                        output_row_3_0 = _mm512_fmadd_pd(a00, b0, output_row_3_0);
                        output_row_3_1 = _mm512_fmadd_pd(a00, b1, output_row_3_1);
                        output_row_3_2 = _mm512_fmadd_pd(a00, b2, output_row_3_2);
                        output_row_3_3 = _mm512_fmadd_pd(a00, b3, output_row_3_3);
                        output_row_3_4 = _mm512_fmadd_pd(a00, b4, output_row_3_4);
                        output_row_3_5 = _mm512_fmadd_pd(a00, b5, output_row_3_5);
                     
                       
                    }
                    _mm512_store_pd(&matrix3[ii][jj], output_row_0_0);
                    _mm512_store_pd(&matrix3[ii][jj + 8], output_row_0_1);
                    _mm512_store_pd(&matrix3[ii][jj + 16], output_row_0_2);
                    _mm512_store_pd(&matrix3[ii][jj + 24], output_row_0_3);
                    _mm512_store_pd(&matrix3[ii][jj + 32], output_row_0_4);
                    _mm512_store_pd(&matrix3[ii][jj + 40], output_row_0_5);

                    _mm512_store_pd(&matrix3[ii + 1][jj], output_row_1_0);
                    _mm512_store_pd(&matrix3[ii + 1][jj + 8], output_row_1_1);
                    _mm512_store_pd(&matrix3[ii + 1][jj + 16], output_row_1_2);
                    _mm512_store_pd(&matrix3[ii + 1][jj + 24], output_row_1_3);
                    _mm512_store_pd(&matrix3[ii + 1][jj + 32], output_row_1_4);
                    _mm512_store_pd(&matrix3[ii + 1][jj + 40], output_row_1_5);

                    _mm512_store_pd(&matrix3[ii + 2][jj], output_row_2_0);
                    _mm512_store_pd(&matrix3[ii + 2][jj + 8], output_row_2_1);
                    _mm512_store_pd(&matrix3[ii + 2][jj + 16], output_row_2_2);
                    _mm512_store_pd(&matrix3[ii + 2][jj + 24], output_row_2_3);
                    _mm512_store_pd(&matrix3[ii + 2][jj + 32], output_row_2_4);
                    _mm512_store_pd(&matrix3[ii + 2][jj + 40], output_row_2_5);

                    _mm512_store_pd(&matrix3[ii + 3][jj], output_row_3_0);
                    _mm512_store_pd(&matrix3[ii + 3][jj + 8], output_row_3_1);
                    _mm512_store_pd(&matrix3[ii + 3][jj + 16], output_row_3_2);
                    _mm512_store_pd(&matrix3[ii + 3][jj + 24], output_row_3_3);
                    _mm512_store_pd(&matrix3[ii + 3][jj + 32], output_row_3_4);
                    _mm512_store_pd(&matrix3[ii + 3][jj + 40], output_row_3_5);


                }              
            }
        }
    }
}
