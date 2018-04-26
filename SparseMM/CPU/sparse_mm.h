#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <string.h>
#include "func.h"
#include <iostream>
#include <map>
#include <omp.h>

using std::vector;
using std::cout;
using std::endl;
using std::map;

class SparseMM {
private:
    CSR *A;
    CSR *B;
    int num_threads;
    
    int A_nnz;
    int A_rank;
    int A_nprocs; 
    vector<int> A_nnz_offsets; 
    vector<int> A_rows_local; 
    vector<int> A_row_ptr;
    vector<int> A_col_idx;
    vector<double> A_value; 
    vector<int> row_nnz_offsets;

    vector<int> B_col_idx;
    vector<int> B_row_ptr;
    vector<double> B_value;
    int B_nnz;
    vector<vector<int> > row_contri;
    vector<map<int, int> > col_to_B_pos_map;
    vector<vector<int> > row_contri_pos;
public:
    SparseMM(CSR *AA, CSR *BB, int num_threadss);

    void GatherData(bool is_first_time);

    int GatherNNZ();

    void Compute0(bool is_first_time);
    
    void Compute1(bool is_first_time);
};
