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
    Config(int dimm, string platformm, int num_threadss, int block_sizee) {
        dim = dimm;
        platform = platformm;
        num_threads = num_threadss;
        block_size = block_sizee; 
    }
    
    int dim;
    int num_threads;
    string platform;
    int block_size;
};


#endif /* config_h */
