//
//  config.h
//  Stencil
//
//  Created by Shi Yu on 2018/1/2.
//  Copyright © 2018年 Shi Yu. All rights reserved.
//

#ifndef config_h
#define config_h

#include <vector>
#include <iostream>
#include <string>

using std::string;

class Config {
private:
public:
    int num_threads;
    int x_dim, y_dim, z_dim;
    int t_steps;
    double alpha, beta_x_0, beta_x_1, beta_y_0, beta_y_1, beta_z_0, beta_z_1;
    
    Config(int num_threadss, int x_dimm, int y_dimm, int z_dimm, int t_stepss, 
            double alphaa, double betaa_x_0, double betaa_x_1, double betaa_y_0, double betaa_y_1, double betaa_z_0, double betaa_z_1) { 
        num_threads = num_threadss;
        x_dim = x_dimm;
        y_dim = y_dimm;
        z_dim = z_dimm; 
        t_steps = t_stepss;
        alpha = alphaa;
        beta_x_0 = betaa_x_0;
        beta_x_1 = betaa_x_1;
        beta_y_0 = betaa_y_0;
        beta_y_1 = betaa_y_1;
        beta_z_0 = betaa_z_0;
        beta_z_1 = betaa_z_1;
    }
};

#endif /* config_h */
