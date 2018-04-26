//
//  config.h
//  MatrixMultiply
//
//  Created by Shi Yu on 2017/12/17.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#ifndef config_h
#define config_h

#include <string>

using std::string;

class Config {
private:
public:
    Config(int dimm, string platformm, int num_threadss, int block_sizee, int stridee, string dir_namee) {
        dim = dimm;
        platform = platformm;
        num_threads = num_threadss; 
        block_size = block_sizee;
        stride = stridee;
        dir_name = dir_namee; 
    }
    
    int dim;
    int num_threads;
    string platform;
    int block_size;
    int stride;
    string dir_name;
};


#endif /* config_h */
