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

    vector<int> B_col_idx;
    vector<int> B_row_ptr;
    vector<double> B_value;
    int B_nnz;
    //vector<vector<int> > row_contri;
    vector<map<int, int> > col_to_B_pos_map;
    vector<vector<int> > row_contri_pos;

    vector<int> row_contri_offsets;
    vector<int> row_contri;
    vector<int> max_cnts;
public:
    SparseMM(CSR *AA, CSR *BB, int num_threadss);

    void Compute0(bool is_first_time);
    
    void Compute1(bool is_first_time);
};
