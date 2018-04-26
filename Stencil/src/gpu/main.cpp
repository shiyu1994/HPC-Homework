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
    dvec64 data_cpu; 
    dvec64 data_gpu;
   
    
    cout << "[********* INIT *********]" << endl;
    cout << "initializing data" << endl;
    double t_start = omp_get_wtime();
    Stencil cpu_stencil(config, data_cpu, 50, 50, 4);
    cpu_stencil.InitData();
    GPUStencil gpu_stencil(config, data_gpu, -1, -1, -1);
    gpu_stencil.InitData(data_cpu); 
    double t_end = omp_get_wtime();
    cout << "initializing data time: " << (t_end - t_start) << endl << endl;
        
    cout << "[********* GPU 2D Blocking *********]" << endl;
    t_start = omp_get_wtime();
    gpu_stencil.Compute();
    t_end = omp_get_wtime();
    cout << endl;
    
    cout << "[********* CPU Naive *********]" << endl;
    t_start = omp_get_wtime();
    cpu_stencil.ComputeNaive();
    t_end = omp_get_wtime();
    cout << "cpu naive time: " << (t_end - t_start) << endl << endl;
     
    cout << "[********* Check *********]" << endl;
    bool check = true;
    //check with naive cpu implementation 
    int z_size = (y_dim + 2) * (x_dim + 2);
    int y_size = x_dim + 2; 
    for(int i = 1; i < z_dim + 1 && check; ++i) {
        for(int j = 1; j < y_dim + 1 && check; ++j) {
           for(int k = 1; k < x_dim + 1; ++k) {
                int index = i * z_size + j * y_size + k;
                if(std::fabs(data_cpu[index] - data_gpu[index]) >= 1e-6) {
                    cout << i << " " << j << " " << k << " " << data_cpu[index] << " " << data_gpu[index] << endl; 
                    check = false;
                    break; 
                }
            }
        }
    }
    cout << "check pass" << endl; 

    return 0;
}
