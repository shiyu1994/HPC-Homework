#include <iostream>
#include "knl_matrix_multiplier.hpp"
#include <vector>
#include "knl_aligned_allocator.h"
#include <omp.h>
#include <x86intrin.h>
#include <xmmintrin.h>
//#include <mkl.h>
#include <cassert>

using std::cout;
using std::endl;
using std::vector;

/*void Check(const dmatrix64 &matrix1, const dmatrix64 &matrix2, const dmatrix64 &matrix3, int n, int num_threads) {
    double *A = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *B = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    double *C = (double *)mkl_malloc( n*n*sizeof( double ), 64 );
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            A[i * n + j] = matrix1[i][j];
            B[i * n + j] = matrix2[i][j];
            C[i * n + j] = 0.0;
        }
    }
    double t_start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n, 1.0, A, n, B, n, 0.0, C, n); 
    double t_end = omp_get_wtime();
    cout << "mkl time: " << (t_end - t_start) << endl;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            if(!(C[i * n + j] - matrix3[i][j] <= 1e-6 && C[i * n + j] - matrix3[i][j] >= -1e-6)) {
                cout << i << " " << j << endl;
                cout << C[i * n + j] << " " << matrix3[i][j] << endl;
            }
            assert(C[i * n + j] - matrix3[i][j] <= 1e-6 && C[i * n + j] - matrix3[i][j] >= -1e-6);
        }
    }

    cout << "check pass" << endl;
        
    MKL_free(A);
    MKL_free(B);
    MKL_free(C);
}*/

int main(int argc, char * argv[]) {
    //double **matrix1, **matrix2, **matrix3; 
    dmatrix64 matrix1, matrix2, matrix3;
    int dim = std::atoi(argv[1]);
    int num_threads = 72; 
    int block_size = 96;
    KNLMatrixMultiplier knlmm(matrix1, matrix2, matrix3, dim, num_threads, block_size);
    double t_start = omp_get_wtime();
    cout << "[********** INIT **********]" << endl;
    cout << "initializing knl data" << endl;
    knlmm.InitData();
    double t_end = omp_get_wtime();
    cout << "knl data intialization time: " << (t_end - t_start) << endl << endl;

    cout << "[********** KNL Compute **********]" << endl;
    t_start = omp_get_wtime();
    knlmm.Compute();
    t_end = omp_get_wtime();
    cout << "knl compute time: " << (t_end - t_start) << endl << endl;

    cout << "[********** Check **********]" << endl;
    //Check(matrix1, matrix2, matrix3, dim, num_threads); 
 
    return 0;
}
