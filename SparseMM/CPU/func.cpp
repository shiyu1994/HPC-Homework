#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mpi.h>
#include <string.h>
#include "func.h"
#include <iostream>
#include <vector>
#include "sparse_ma.h"
#include "sparse_mm.h"

using std::cout;
using std::endl;
using std::vector;

extern int output(int rows, int *row_ptr, int *col_idx, double *val);
extern void Qsort(int *a, double *b, int low, int high);

SparseMA *sparse_ma = NULL;

/* B = AT+A */
int sparse_AT_plus_A_type0(CSR *B, CSR *A, bool first_time)
{
    //SparseMA *sparse_ma = NULL;
    if(first_time) {
        sparse_ma = new SparseMA(A, B, 12);
    }
    sparse_ma->Compute0(first_time); 
    return 0;
}


/* B = AT+A */
int sparse_AT_plus_A_type1(CSR *B, CSR *A, bool first_time)
{
    if(first_time) {
        sparse_ma = new SparseMA(A, B, 12);
    }
    sparse_ma->Compute1(first_time); 
    return 0;
}

SparseMM *sparse_mm = NULL;

/* B = AT*A */
int sparse_ATA_type0(CSR *B, CSR *A, bool first_time)
{
    if(first_time) {
        sparse_mm = new SparseMM(A, B, 24); 
    }
    sparse_mm->Compute0(first_time);
    return 0;
}

/* B = AT*A */
int sparse_ATA_type1(CSR *B, CSR *A, bool first_time)
{
    if(first_time) {
        sparse_mm = new SparseMM(A, B, 24); 
    }
    sparse_mm->Compute1(first_time);

    return 0;
}

