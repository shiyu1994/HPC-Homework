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
    vector<int> rows_offset;
    vector<int> rows_local; 
    vector<int> rows;
    vector<int> col_idx;
    vector<double> values;
    int num_threads;
    vector<double*> addrs;
    vector<double*> addr_ts;
    
    vector<int> col_to_row_map1;  
    vector<double> col_to_row_map2;  

    vector<int> col_to_row_map_size;
    vector<int> col_to_row_map_cnt;

    
    void PartitionTasks();

    void GatherData();

    void ComputeStandAlone();

    void ComputeStandAlonePreprocess(bool is_first_time);

public:
    SparseMA(CSR *AA, CSR *BB, int num_threadss);

    void Compute0(bool is_first_time); 
    void Compute1(bool is_first_time);
};
