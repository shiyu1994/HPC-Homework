//
//  main.cpp
//  GPUMatrixMultiplier
//
//  Created by Shi Yu on 2017/12/23.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#include <iostream>
#include "gpu_matrix_multiplier.hpp"
#include <vector>
#include "aligned_allocator.h"
#include <omp.h>
#include <cassert>
//#include <mkl.h>

using std::vector;
using std::cout;
using std::endl;

/*void Check(const dvec64 &matrix1, const dvec64 &matrix2, const dvec64 &matrix3, int n, int num_threads) {
    double *A = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *B = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *C = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            A[i * n + j] = matrix1[i * n + j];
            B[i * n + j] = matrix2[i * n + j];
            C[i * n + j] = 0.0;
        }
    }
    double t_start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0, A, n, B, n, 0.0, C, n); 
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(!(C[i * n + j] - matrix3[i * n + j] <= 1e-6 && C[i * n + j] - matrix3[i * n + j] >= -1e-6)) {
                cout << i << " " << j << endl;
                cout << C[i * n + j] << " " << matrix3[i * n + j] << endl;
            }
            assert(C[i * n + j] - matrix3[i * n + j] <= 1e-6 && C[i * n + j] - matrix3[i * n + j] >= -1e-6);
        }
    }
    double t_end = omp_get_wtime();

    cout << "mkl time: " << (t_end - t_start) << endl;
    cout << "check pass" << endl;
        
    MKL_free(A);
    MKL_free(B);
    MKL_free(C);
}*/

int main(int argc, const char * argv[]) {
    int dim = std::atoi(argv[1]);
    int num_threads = 24;
    dvec64 matrix1(dim * dim), matrix2(dim * dim), matrix3(dim * dim); 
    double t_start = omp_get_wtime();
    cout << "[********** INIT **********]" << endl;
    cout << "initializing data" << endl;
#pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 0; i < dim * dim; ++i) {
        matrix1[i] = matrix2[i] = std::rand() * 1.0 / RAND_MAX; 
    } 
    double t_end = omp_get_wtime();
    cout << "initalize time: " << (t_end - t_start) << endl << endl;

    cout << "[********** Compute GPU **********]" << endl;
    GPUMatrixMultiplier gpumm(matrix1, matrix2, matrix3, dim);
    gpumm.Compute(); 
    cout << endl;
    
    //cout << "[********** Check **********]" << endl;
    //Check(matrix1, matrix2, matrix3, dim, num_threads); 
    return 0;
}
