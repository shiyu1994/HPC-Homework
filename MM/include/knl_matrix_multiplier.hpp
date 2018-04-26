#include <iostream>
#include <vector>
#include "knl_aligned_allocator.h"

using std::vector;
using std::cout;
using std::endl;

class KNLMatrixMultiplier {
private:
    /*double **matrix1;
    double **matrix2;
    double **matrix3;*/
    dmatrix64 &matrix1;
    dmatrix64 &matrix2;
    dmatrix64 &matrix3;
    int dim;
    int padded_dim;
    int num_threads;
    int block_size;


    inline int CalcRowBlockID(int row_nblocks, int col_nblocks, int row_blocks_per_threads, int row_remainder,int block_id) {
        if(block_id < col_nblocks * (row_nblocks / row_blocks_per_threads) * row_blocks_per_threads) {
           
            return block_id % row_blocks_per_threads + block_id / (col_nblocks * row_blocks_per_threads) * row_blocks_per_threads;
          
        }
        else {
         
            return row_nblocks - row_remainder + block_id % (col_nblocks * row_blocks_per_threads) % row_remainder;
        
        }
    }
        
    inline int CalcColBlockID(int row_nblocks, int col_nblocks, int row_blocks_per_threads, int row_remainder, int block_id) {
        if(block_id < col_nblocks * (row_nblocks / row_blocks_per_threads) * row_blocks_per_threads) {
            return block_id % (col_nblocks * row_blocks_per_threads) / row_blocks_per_threads;
        }
        else {
            return block_id % (col_nblocks * row_blocks_per_threads) / row_remainder;
        }
    }
public:
    //KNLMatrixMultiplier(double **matrixx1, double **matrixx2, double **matrixx3, int dimm, int num_threads, int block_sizee);
    KNLMatrixMultiplier(dmatrix64 &matrixx1, dmatrix64 &matrixx2, dmatrix64 &matrixx3, int dimm, int num_threads, int block_sizee);


    void InitData(); 

    void Compute();
};
