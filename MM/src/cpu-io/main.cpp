//
//  main.cpp
//  MatrixMultiply
//
//  Created by Shi Yu on 2017/10/16.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#include <iostream>
#include <vector>                   
#include <omp.h>
#include <cstdlib>
#include <cfloat>
#include <cassert>
//#include <mkl.h>
#include "aligned_allocator.h"
#include <x86intrin.h>
#include "cpu_io_config.h"
#include "cpu_io_matrix_multiplier.hpp"
#include <string>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cmath>

using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::ofstream;
using std::ifstream;

/*void Check(string fname_A, string fname_B, string result, int dim) {
    ifstream finA(fname_A);
    ifstream finB(fname_B);
    ifstream finC(result);
    
    double *A = (double *)mkl_malloc( dim*dim*sizeof( double ), 64 );
    double *B = (double *)mkl_malloc( dim*dim*sizeof( double ), 64 );
    double *C = (double *)mkl_malloc( dim*dim*sizeof( double ), 64 );
    
    dmatrix64 result_C(dim);
    for(int i = 0; i < dim; ++i) {
        result_C[i].resize(dim, 0.0);
    }
    
    //prepare data for mkl
    for(int i = 0; i < dim; ++i) {
        for(int j = 0; j < dim; ++j) {
            finA >> A[i * dim + j];
            finB >> B[i * dim + j];
            finC >> result_C[i][j];
            C[i * dim + j] = 0.0;
        }
    }
    double time_start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, dim, dim, 1.0, A, dim, B, dim, 0.0, C, dim);
    double time_end = omp_get_wtime();
    cout << "mkl time: " << (time_end - time_start) << endl;            
    for(int i = 0; i < dim; ++i) {
        for(int j = 0; j < dim; ++j) {
            if(std::fabs(C[i * dim + j] - result_C[i][j]) >= 1e-5) {
                cout << i << " " << j << endl;
                cout << C[i * dim + j] << " " << result_C[i][j] << endl;
            }
            assert(std::fabs(C[i * dim + j] - result_C[i][j]) < 1e-5);
        }
    }
    
    cout << "check pass" << endl;
    
    MKL_free(A);
    MKL_free(B);
    MKL_free(C);
}*/

void GenData(string fname_A, string fname_B, string fname_C, int dim) {
    std::srand(static_cast<int>(std::time(nullptr)));
    ofstream foutA(fname_A);
    ofstream foutB(fname_B);
    ofstream foutC(fname_C);
    for(int i = 0; i < dim; ++i) {
        for(int j = 0; j < dim; ++j) {
            foutA << (std::rand() * 1.0 / RAND_MAX) << " ";
            foutB << (std::rand() * 1.0 / RAND_MAX) << " ";
            foutC << 0.0 << " ";
        }
        foutA << endl;
        foutB << endl;
        foutC << endl;
    }
}

int main(int argc, char * argv[]) {
    int n = std::atoi(argv[1]);
    int num_threads = 24;    
    string dir_name = argv[2]; 
    string fname_A = dir_name + "/A";
    string fname_B = dir_name + "/B";
    string fname_C = dir_name + "/C";
    
    cout << "[********** INIT **********]" << endl;
    cout << "initializing data, writing to file: " << fname_A << " and " << fname_B << endl;
    double t_start = omp_get_wtime();
    GenData(fname_A, fname_B, fname_C, n);
    double t_end = omp_get_wtime();
    cout << "initializing time: " << (t_end - t_start) << endl << endl;

    Config conf(n, "cpu", num_threads, 96, 192, dir_name);
    MatrixMultiplier mm(fname_A, fname_B, fname_C, conf);
    
    cout << "[********** Compute CPU-IO **********]" << endl;
    t_start = omp_get_wtime();
    string result = mm.Compute();
    t_end = omp_get_wtime();
    cout << "io matrix multiplication time: " << (t_end - t_start) << endl;
    cout << "finished, result stored in file " << result << endl << endl;
    
    //cout << "[********** Check **********]" << endl;
    //Check(fname_A, fname_B, result, n); 
    
    return 0;
}
