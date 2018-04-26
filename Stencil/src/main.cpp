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
    int x_dim = std::atoi(argv[2]);
    int y_dim = std::atoi(argv[3]);
    int z_dim = std::atoi(argv[4]);
    int t_steps = std::atoi(argv[5]);
    Config config(argv[1], 24, x_dim, y_dim, z_dim, t_steps, 0.7, 0.06, 0.04, 0.07, 0.03, 0.08, 0.02);
    dvec64 data; 
    dvec64 data2;
   
    if(config.platform == "cpu") {
    Stencil stencil(config, data, 50, 50, 4);
    stencil.InitData();

    Stencil stencil2(config, data2, 50, 50, 4);
    stencil2.InitData(data);
    double t_start = omp_get_wtime();
    stencil2.ComputeNaive();
    double t_end = omp_get_wtime();
    cout << "cpu naive compute time " << (t_end - t_start) << endl;

    t_start = omp_get_wtime();
    stencil.Compute();
    t_end = omp_get_wtime();
    cout << "cpu 3.5d blocking compute time " << (t_end - t_start) << endl;
    //check with naive cpu implementation
        if(std::atoi(argv[6]) == 1) {
        int z_size = (y_dim + 2) * (x_dim + 2);
        int y_size = x_dim + 2;
        bool check = true;
        for(int i = 1; i < z_dim + 1 && check; ++i) {
            for(int j = 1; j < y_dim + 1 && check; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    if(std::abs(data[index] - data2[index]) >= 1e-6) {
                        cout << i << " " << j << " " << k << " " << data[index] << " " << data2[index] << endl;
                        check = false;
                       break; 
                    }
                }
            }
        }
        }
        
        cout << "check pass" << endl;
    }
    else if(config.platform == "gpu") {
        dvec64 data2;
        Stencil cpu_stencil(config, data2, 50, 50, 4);
        cpu_stencil.InitData();
        GPUStencil stencil(config, data, -1, -1, -1);
        stencil.InitData(data2); 
        
        stencil.Compute();
        //stencil.PrintResult();
        
        //dvec64 data2;
        //Stencil cpu_stencil(config, data2, 50, 50, 4);
        //cpu_stencil.InitData();
        double t_start = omp_get_wtime();
        cpu_stencil.ComputeNaive();
        double t_end = omp_get_wtime();
        cout << "cpu naive time: " << (t_end - t_start) << endl;
     
        //check with naive cpu implementation
        if(std::atoi(argv[6]) == 1) {
        int z_size = (y_dim + 2) * (x_dim + 2);
        int y_size = x_dim + 2;
        bool check = true;
        for(int i = 1; i < z_dim + 1 && check; ++i) {
            for(int j = 1; j < y_dim + 1 && check; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    if(std::abs(data[index] - data2[index]) >= 1e-6) {
                        cout << i << " " << j << " " << k << " " << data[index] << " " << data2[index] << endl;
                        check = false;
                       break; 
                    }
                }
            }
        }
        
        cout << "check pass" << endl;
        }
    }
    else if(config.platform == "gpu-cpu") {
        dvec64 data2;
        Stencil cpu_stencil(config, data2, 50, 50, 10);
        cpu_stencil.InitData();
        GPUCPUStencil stencil(config, data, -1, -1, 6);
        stencil.InitData(data2); 
        
        stencil.Compute();
        //stencil.PrintResult();
        
        //dvec64 data2;
        //Stencil cpu_stencil(config, data2, 50, 50, 4);
        //cpu_stencil.InitData();
        double t_start = omp_get_wtime();
        cpu_stencil.ComputeNaive();
        double t_end = omp_get_wtime();
        cout << "cpu naive time: " << (t_end - t_start) << endl;
     
        //check with naive cpu implementation
        if(std::atoi(argv[6]) == 1) {
        int z_size = (y_dim + 2) * (x_dim + 2);
        int y_size = x_dim + 2;
        bool check = true;
        for(int i = 1; i < z_dim + 1 && check; ++i) {
            for(int j = 1; j < y_dim + 1 && check; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    if(std::abs(data[index] - data2[index]) >= 1e-6) {
                        cout << i << " " << j << " " << k << " " << data[index] << " " << data2[index] << endl;
                        check = false;
                        break; 
                    }
                }
            }
        }
        
        cout << "check pass" << endl;
        }
    }
    else if(config.platform == "cpu-queue") { 
        //MPI_Init(&argc, &argv);
        dvec64 data_cpu_naive;
        dvec64 data_cpu_queue;
        CPUQueueStencil cpu_queue_stencil(config, data_cpu_queue, 25, 25, 4, &argc, &argv);
        Stencil cpu_stencil(config, data_cpu_naive, 25, 25, 4);
        cpu_queue_stencil.InitData();
        int rank = cpu_queue_stencil.InitMPI();
        if(rank == 0) {
            cpu_stencil.InitData(data_cpu_queue);
        }
        cpu_queue_stencil.Compute();
        if(rank == 0) {
            cpu_stencil.ComputeNaive();
            if(std::atoi(argv[6]) == 1) {
                int z_size = (y_dim + 2) * (x_dim + 2);
                int y_size = x_dim + 2;
                bool check = true;
                for(int i = 1; i < z_dim + 1 && check; ++i) {
                    for(int j = 1; j < y_dim + 1 && check; ++j) {
                        for(int k = 1; k < x_dim + 1; ++k) {
                            int index = i * z_size + j * y_size + k;
                            if(std::abs(data[index] - data2[index]) >= 1e-6) {
                                cout << i << " " << j << " " << k << " " << data[index] << " " << data2[index] << endl;
                                check = false;
                                break; 
                            }
                        }
                    }
                }
                cout << "check pass" << endl;
            }
        }
        cpu_queue_stencil.FinalizeMPI();
    }

    return 0;
}
