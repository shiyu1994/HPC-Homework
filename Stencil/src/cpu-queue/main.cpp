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
   
    dvec64 data_cpu_queue, data_cpu_naive;
    
    double t_start, t_end;
     
    t_start = omp_get_wtime();
    int block_x = 50, block_y = 50;
    if(x_dim <= 100 || y_dim <= 100) {
        block_x = 20;
        block_y = 20;
    }
    CPUQueueStencil cpu_queue_stencil(config, data_cpu_queue, block_x, block_y, 5, &argc, &argv); 
    Stencil cpu_stencil(config, data_cpu_naive, 25, 25, 4);
    int rank = cpu_queue_stencil.InitMPI();
    if(rank == 0) {
        cout << "[********** INIT **********]" << endl;
        cout << "initializing data" << endl;
    }
    cpu_queue_stencil.InitData(); 
    if(rank == 0) {
        cpu_stencil.InitData(data_cpu_queue);
        t_end = omp_get_wtime();
        cout << "initializing data time: " << (t_end - t_start) << endl << endl;
    } 

    if(rank == 0) {
        cout << "[********** CPU-Queue 3.5D Blocking **********]" << endl;
    }
    
    cpu_queue_stencil.Compute();
   
    if(rank == 0) {
        cout << "[********** Check **********]" << endl;
        double t_start = omp_get_wtime();
        cpu_stencil.ComputeNaive();
        double t_end = omp_get_wtime();
        cout << "cpu naive time: " << (t_end - t_start) << endl;
       
            int z_size = (y_dim + 2) * (x_dim + 2);
            int y_size = x_dim + 2;
            bool check = true;
            for(int i = 1; i < z_dim + 1 && check; ++i) {
                for(int j = 1; j < y_dim + 1 && check; ++j) {
                    for(int k = 1; k < x_dim + 1; ++k) {
                        int index = i * z_size + j * y_size + k;
                        if(std::fabs(data_cpu_queue[index] - data_cpu_naive[index]) >= 1e-6) {
                            cout << i << " " << j << " " << k << " " << data_cpu_queue[index] << " " << data_cpu_naive[index] << endl;
                            check = false;
                            break; 
                        }
                    }
                }
            }
            cout << "check pass" << endl << endl;
       
    }
    cpu_queue_stencil.FinalizeMPI(); 

    return 0;
}
