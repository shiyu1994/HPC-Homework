#ifndef _FUNC_H_
#define _FUNC_H_

typedef struct
{
    int rank; //My rank ID
    int nprocs; //The number of processes
    int rows, cols;
    int nnz; // The total number of non-zeros
    int rows_local; // The local number of rows
    int rows_offset; // The offset of rows
    int nnz_local; //The local number of non-zeros
    int *row_ptr;
    int *col_idx;
    double *value;
} CSR;

int sparse_AT_plus_A_type0(CSR *B, CSR *A, bool first_time);
int sparse_AT_plus_A_type1(CSR *B, CSR *A, bool first_time);

int sparse_ATA_type0(CSR *B, CSR *A, bool first_time);
int sparse_ATA_type1(CSR *B, CSR *A, bool first_time);

int check_sparse_AT_plus_A(CSR, CSR);
int check_sparse_ATA(CSR, CSR);

#endif
