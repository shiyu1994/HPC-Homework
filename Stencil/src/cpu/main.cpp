//
//  main.cpp
//  Stencil
//
//  Created by Shi Yu on 2018/1/2.
//  Copyright © 2018年 Shi Yu. All rights reserved.
//

#include <iostream>
#include "config.h"
#include <omp.h> 
#include "aligned_allocator.h"
#include "stencil.hpp"
#include <cmath>

using std::cout;
using std::endl;

int main(int argc, char * argv[]) {
    int x_dim = std::atoi(argv[1]);
    int y_dim = std::atoi(argv[2]);
    int z_dim = std::atoi(argv[3]);
    int t_steps = std::atoi(argv[4]);
    Config config(24, x_dim, y_dim, z_dim, t_steps, 0.7, 0.06, 0.04, 0.07, 0.03, 0.08, 0.02);
    dvec64 data_3_5_D; 
    dvec64 data_naive;
   
    cout << endl;
    cout << "[**********INIT**********]" << endl;
    cout << "initializing data" << endl;
    double t_start = omp_get_wtime();
    int block_size_x = 50, block_size_y = 50;
    if(x_dim <= 100) {
        block_size_x = 20;
    }
    if(y_dim <= 100) {
        block_size_y = 20;
    }
    Stencil stencil_3_5_D(config, data_3_5_D, block_size_x, block_size_y, 4);
    stencil_3_5_D.InitData();

    Stencil stencil_naive(config, data_naive, block_size_x, block_size_y, 4);
    stencil_naive.InitData(data_3_5_D);
    double t_end = omp_get_wtime();
    cout << "initalizing data time: " << (t_end - t_start) << endl << endl;

    cout << "[********** Compute 3.5D Blocking **********]" << endl;
    t_start = omp_get_wtime();
    stencil_3_5_D.Compute();
    t_end = omp_get_wtime();
    cout << "cpu 3.5d blocking compute time " << (t_end - t_start) << endl << endl;


    cout << "[********** Compute Naive **********]" << endl;
    t_start = omp_get_wtime();
    stencil_naive.ComputeNaive();
    t_end = omp_get_wtime();
    cout << "cpu naive compute time " << (t_end - t_start) << endl << endl;
 
    cout << "[********** Check **********]" << endl;
    //check with naive cpu implementation 
    int z_size = (y_dim + 2) * (x_dim + 2);
    int y_size = x_dim + 2;
    bool check = true;
    for(int i = 1; i < z_dim + 1 && check; ++i) {
        for(int j = 1; j < y_dim + 1 && check; ++j) {
            for(int k = 1; k < x_dim + 1; ++k) {
                int index = i * z_size + j * y_size + k;
                if(std::fabs(data_3_5_D[index] - data_naive[index]) >= 1e-6) {
                    cout << i << " " << j << " " << k << " " << data_3_5_D[index] << " " << data_naive[index] << endl;
                    check = false;
                    break; 
                }
            }
        }
    }
        
    cout << "check pass" << endl << endl;

    
    return 0;
}
