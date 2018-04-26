//
//  gpu_matrix_multiplier.hpp
//  GPUMatrixMultiplier
//
//  Created by Shi Yu on 2017/12/23.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#ifndef gpu_matrix_multiplier_hpp
#define gpu_matrix_multiplier_hpp

#include <stdio.h>
#include "aligned_allocator.h"

class GPUMatrixMultiplier {
private:
    //true dimension of matrix
    int dim;
    //padded dimension of matrix
    int dim_pad; 
    //reference to cpu matrix data, stored in vector
    const dvec64 &matrix1;
    const dvec64 &matrix2;
    dvec64 &matrix3;
    
    //cuda memory pointers
    double *cuda_matrix1, *cuda_matrix2, *cuda_matrix3;
    
    void AllocGPUMem();
    
    void FreeGPUMem(); 
    
    void CopyDataFromCPUToGPU();
    
    void CopyResultFromGPUToCPU();
    
    void CalcGridAndBlockDim(int &grid_dim_x, int &grid_dim_y, int &block_dim_x, int &block_dim_y);
    
    void PrintMatrix(const dvec64 &matrix, int dim);
public:
    GPUMatrixMultiplier(const dvec64 &matrixx1, const dvec64 &matrixx2, dvec64 &matrixx3, int dim);
    
    void Compute();  
};

#endif /* gpu_matrix_multiplier_hpp */
