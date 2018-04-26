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
#include "cpu_io_config.h"
#include <string>

using std::string; 

class MatrixMultiplier {
private:
    dmatrix64 matrix1;
    dmatrix64 matrix2;
    dmatrix64 matrix3;
    Config config;
    
    string fname_A, fname_B, fname_C;   
    
    int padded_dim;
    
    void ComputeStandAlone(int a_row_begin, int a_col_begin, int b_row_begin, int b_col_begin);
    
    void ComputeMPI();
    
    void ComputeGPU();
    
    void ComputeKernel(int row_size, int dimm, int col_size);
    
    void StandAloneInitData(); 
    
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
    
    void ReadLinesIntoMemory(string fname, int lines_begin, int lines_end,
                             dmatrix64 &matrix, int padded_rows, int padded_cols);
    
    void WriteOut(string fname, int rows, int cols, dmatrix64 &matrix); 
    
public:
    MatrixMultiplier(string fname_A, string fname_B, string fname_C, Config configg);   
    
    string Compute();
    
    //check the correctness using MKL
    void Check();
}; 

#endif /* matrix_multiplier_hpp */
