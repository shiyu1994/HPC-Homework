#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <string.h>
#include "func.h"

using std::vector;
using std::cout;
using std::endl;

class SparseMA {
protected:
    CSR *A;
    CSR *B;
    int A_nnz;
    int A_rank;
    int A_nprocs; 
    vector<int> A_nnz_offsets; 
    vector<int> A_rows_local; 
    vector<int> A_row_ptr;
    vector<int> A_col_idx;
    vector<double> A_value;
    int num_threads;
    vector<int> row_nnz_offsets;
    vector<double*> addrs;
    vector<double*> addr_ts; 

    void GatherData(bool is_first_time);
    int GatherNNZ();
 
    void ComputeWithPreprocess(bool is_first_time);

public:
    SparseMA(CSR *AA, CSR *BB, int num_threadss);

    void Compute0(bool is_first_time); 
    void Compute1(bool is_first_time); 
};
