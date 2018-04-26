//
//  matrix_multiplier.hpp
//  MatrixMultiply
//
//  Created by Shi Yu on 2017/12/17.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#ifndef matrix_multiplier_hpp
#define matrix_multiplier_hpp

#include <stdio.h>
#include "aligned_allocator.h"
#include "config.h" 
#include <ctime>

class MatrixMultiplier {
private:
    dmatrix64 &matrix1;
    dmatrix64 &matrix2;
    dmatrix64 &matrix3;
    Config config;
    
    int padded_dim;
    
    void ComputeStandAlone();
    
    void ComputeMPI();
    
    void ComputeGPU();
    
    void ComputeKernel(int row_size, int dimm, int col_size); 
    
    inline int CalcRowBlockID(int row_nblocks, int col_nblocks,
                              int row_blocks_per_threads, 
                              int row_remainder,
                              int block_id) {
        if(block_id < col_nblocks * (row_nblocks / row_blocks_per_threads) * row_blocks_per_threads) {
            return block_id % row_blocks_per_threads +
                block_id / (col_nblocks * row_blocks_per_threads) * row_blocks_per_threads;
        }
        else {
            return row_nblocks - row_remainder + block_id % (col_nblocks * row_blocks_per_threads) % row_remainder;
        }
    }
    
    inline int CalcColBlockID(int row_nblocks, int col_nblocks,
                              int row_blocks_per_threads,
                              int row_remainder,                    
                              int block_id) {
        if(block_id < col_nblocks * (row_nblocks / row_blocks_per_threads) * row_blocks_per_threads) {
            return block_id % (col_nblocks * row_blocks_per_threads) / row_blocks_per_threads;
        }
        else {
            return block_id % (col_nblocks * row_blocks_per_threads) / row_remainder;
        }
    }
    
public:
    MatrixMultiplier(dmatrix64 &matrixx1, dmatrix64 &matrixx2, dmatrix64 &matrixx3, Config configg);
    
    void InitData(); 

    void Compute();
    
    //check the correctness using MKL
    void Check();
}; 

#endif /* matrix_multiplier_hpp */
