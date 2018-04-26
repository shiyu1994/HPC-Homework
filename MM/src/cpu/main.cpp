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
#include "config.h"
#include "matrix_multiplier.hpp"

using std::vector;
using std::cout;
using std::endl;

int main(int argc, char * argv[]) {
    int n = std::atoi(argv[1]);   
    int num_threads = 24;
    vector<vector<double, AlignmentAllocator<double, 64>>> matrix1, matrix2, matrix3;
    
    Config conf(n, "cpu", num_threads, 96);     
    MatrixMultiplier mm(matrix1, matrix2, matrix3, conf);
    
    cout << "[********** INIT **********]" << endl;
    cout << "initializing data" << endl;
    double t_start_init = omp_get_wtime();
    mm.InitData();
    double t_end_init = omp_get_wtime();
    cout << "init data time: " << (t_end_init - t_start_init) << endl << endl;
    
    cout << "[********** CPU Computing **********]" << endl;
    double t_start_comp = omp_get_wtime();
    mm.Compute();
    double t_end_comp = omp_get_wtime();
    cout << "cpu computing time: " << (t_end_comp - t_start_comp) << endl << endl;
    
    //cout << "[********** Check **********]" << endl;
    //mm.Check();  
    
    return 0;
}
