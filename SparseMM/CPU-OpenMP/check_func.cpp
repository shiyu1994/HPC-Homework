#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include "func.h"

#define LL int 
void Qsort(LL *a, double *b, LL low, LL high)
{
  if(low >= high)
  {
    return;
  }
  LL first = low;
  LL last = high;
  LL key = a[first];/*用字表的第一个记录作为枢轴*/
  double bkey = b[first];

  while(first < last)
  {
    while(first < last && a[last] >= key)
    {
      --last;
    }

    a[first] = a[last];/*将比第一个小的移到低端*/
    b[first] = b[last];

    while(first < last && a[first] <= key)
    {
      ++first;
    }

    a[last] = a[first];    
    b[last] = b[first];
    /*将比第一个大的移到高端*/
  }
  a[first] = key;/*枢轴记录到位*/
  b[first] = bkey;
  Qsort(a, b, low, first-1);
  Qsort(a, b, first+1, high);
}
int check_sparse_AT_plus_A_kernel(CSR *B, CSR *A)
{
    int rows =  A->rows;
    int cols =  A->cols;
    int nnz  =  A->nnz;
    int rank = A->rank;
    int nprocs = A->nprocs;

    if(rank == 0)
    {
        int *rows_per_proc = (int*)malloc(nprocs*sizeof(int));
        int *row_ptr  = (int*)malloc((rows+1)*sizeof(int));
        int *pointerE = NULL;//(int*)malloc(rows*sizeof(int));
        int *col_idx  = (int*)malloc(nnz*sizeof(int));
        double *value = (double*)malloc(nnz*sizeof(double));
        memcpy(row_ptr, A->row_ptr, (A->rows_local+1)*sizeof(int));
        memcpy(col_idx, A->col_idx, A->nnz_local*sizeof(int));
        memcpy(value,   A->value,   A->nnz_local*sizeof(double));
        int rows_offset = A->rows_local;
        int nnz_offset = A->nnz_local;
        rows_per_proc[0] = A->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local;
            int nnz_local;
            MPI_Status status;
            MPI_Recv(&rows_local, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&nnz_local,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(col_idx + nnz_offset,  nnz_local,    MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(value   + nnz_offset,  nnz_local, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
            rows_offset += rows_local;
            nnz_offset += nnz_local;
            rows_per_proc[i] = rows_local;
        }
        sparse_matrix_t AA;
        sparse_matrix_t BB;
        mkl_sparse_d_create_csr(&AA, SPARSE_INDEX_BASE_ZERO, rows, cols, row_ptr, row_ptr+1, col_idx, value);
        mkl_sparse_d_add(SPARSE_OPERATION_TRANSPOSE, AA, 1.0, AA, &BB);
        free(row_ptr); row_ptr = NULL;
        free(col_idx); col_idx = NULL;
        free(value);   value   = NULL;

        sparse_index_base_t type;
        mkl_sparse_d_export_csr(BB, &type, &rows, &cols, &row_ptr, &pointerE, &col_idx, &value); 

        B->rank = rank;
        B->nprocs = nprocs;
        B->rows = rows;
        B->cols = cols;
        B->nnz = row_ptr[rows] - row_ptr[0];
        B->rows_local  = rows_per_proc[0];
        B->rows_offset = 0;
        B->nnz_local   = row_ptr[B->rows_local] - row_ptr[0];
        B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
        B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
        B->value = (double*)malloc(B->nnz_local*sizeof(double));
        rows_offset = B->rows_local;
        int total_nnz = row_ptr[rows] - row_ptr[0];
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            int nnz_local = row_ptr[rows_offset+rows_local] - row_ptr[rows_offset];
            MPI_Send(&total_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_local, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&rows_offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(&nnz_local, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }
        memcpy(B->row_ptr, row_ptr, (rows_per_proc[0]+1)*sizeof(int));
        memcpy(B->col_idx, col_idx, (row_ptr[B->rows_local]-row_ptr[0])*sizeof(int));
        memcpy(B->value,   value,   (row_ptr[B->rows_local]-row_ptr[0])*sizeof(double));
        rows_offset = B->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            MPI_Send(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(col_idx + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_INT,    i, 5, MPI_COMM_WORLD);
            MPI_Send(value   + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_DOUBLE, i, 6, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }

        mkl_sparse_destroy(AA);
        mkl_sparse_destroy(BB);
        free(rows_per_proc);
    }
    else
    {
        MPI_Status status;
        MPI_Send(&A->rows_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&A->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(A->row_ptr, A->rows_local+1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(A->col_idx, A->nnz_local,    MPI_INT, 0, 3, MPI_COMM_WORLD);
        MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        B->rank = rank;
        B->nprocs = nprocs;
        B->rows = rows;
        B->cols = cols;
        MPI_Recv(&B->nnz,         1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&B->rows_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&B->rows_offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&B->nnz_local,   1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
        B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
        B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
        B->value = (double*)malloc(B->nnz_local*sizeof(double));
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

int check_sparse_ATA_kernel(CSR *B, CSR *A)
{
    int rows =  A->rows;
    int cols =  A->cols;
    int nnz  =  A->nnz;
    int rank = A->rank;
    int nprocs = A->nprocs;

    if(rank == 0)
    {
        int *rows_per_proc = (int*)malloc(nprocs*sizeof(int));
        int *row_ptr  = (int*)malloc((rows+1)*sizeof(int));
        int *pointerE = NULL;//(int*)malloc(rows*sizeof(int));
        int *col_idx  = (int*)malloc(nnz*sizeof(int));
        double *value = (double*)malloc(nnz*sizeof(double));
        memcpy(row_ptr, A->row_ptr, (A->rows_local+1)*sizeof(int));
        memcpy(col_idx, A->col_idx, A->nnz_local*sizeof(int));
        memcpy(value,   A->value,   A->nnz_local*sizeof(double));
        int rows_offset = A->rows_local;
        int nnz_offset = A->nnz_local;
        rows_per_proc[0] = A->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local;
            int nnz_local;
            MPI_Status status;
            MPI_Recv(&rows_local, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&nnz_local,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            MPI_Recv(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(col_idx + nnz_offset,  nnz_local,    MPI_INT, i, 3, MPI_COMM_WORLD, &status);
            MPI_Recv(value   + nnz_offset,  nnz_local, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
            rows_offset += rows_local;
            nnz_offset += nnz_local;
            rows_per_proc[i] = rows_local;
        }
        sparse_matrix_t AA;
        sparse_matrix_t BB;
        mkl_sparse_d_create_csr(&AA, SPARSE_INDEX_BASE_ZERO, rows, cols, row_ptr, row_ptr+1, col_idx, value);
        mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE, AA, AA, &BB);
        free(row_ptr); row_ptr = NULL;
        free(col_idx); col_idx = NULL;
        free(value);   value   = NULL;

        sparse_index_base_t type;
        mkl_sparse_d_export_csr(BB, &type, &rows, &cols, &row_ptr, &pointerE, &col_idx, &value); 

        {
          int i, j;
          for ( i = 0; i < rows; i ++ ) {
            Qsort(col_idx, value, row_ptr[i], row_ptr[i+1]-1);
          }
        }

        B->rank = rank;
        B->nprocs = nprocs;
        B->rows = rows;
        B->cols = cols;
        B->nnz = row_ptr[rows] - row_ptr[0];
        B->rows_local  = rows_per_proc[0];
        B->rows_offset = 0;
        B->nnz_local   = row_ptr[B->rows_local] - row_ptr[0];
        B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
        B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
        B->value = (double*)malloc(B->nnz_local*sizeof(double));
        rows_offset = B->rows_local;
        int total_nnz = row_ptr[rows] - row_ptr[0];
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            int nnz_local = row_ptr[rows_offset+rows_local] - row_ptr[rows_offset];
            MPI_Send(&total_nnz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&rows_local, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&rows_offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(&nnz_local, 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }
        memcpy(B->row_ptr, row_ptr, (rows_per_proc[0]+1)*sizeof(int));
        memcpy(B->col_idx, col_idx, (row_ptr[B->rows_local]-row_ptr[0])*sizeof(int));
        memcpy(B->value,   value,   (row_ptr[B->rows_local]-row_ptr[0])*sizeof(double));
        rows_offset = B->rows_local;
        for(int i = 1; i < nprocs; i ++)
        {
            int rows_local = rows_per_proc[i];
            MPI_Send(row_ptr + rows_offset, rows_local+1, MPI_INT, i, 4, MPI_COMM_WORLD);
            MPI_Send(col_idx + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_INT,    i, 5, MPI_COMM_WORLD);
            MPI_Send(value   + row_ptr[rows_offset], row_ptr[rows_offset+rows_local]-row_ptr[rows_offset], MPI_DOUBLE, i, 6, MPI_COMM_WORLD);
            rows_offset += rows_local;
        }

        mkl_sparse_destroy(AA);
        mkl_sparse_destroy(BB);
        free(rows_per_proc);
    }
    else
    {
        MPI_Status status;
        MPI_Send(&A->rows_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&A->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(A->row_ptr, A->rows_local+1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(A->col_idx, A->nnz_local,    MPI_INT, 0, 3, MPI_COMM_WORLD);
        MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
        B->rank = rank;
        B->nprocs = nprocs;
        B->rows = rows;
        B->cols = cols;
        MPI_Recv(&B->nnz,         1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&B->rows_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&B->rows_offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(&B->nnz_local,   1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
        B->row_ptr = (int*)malloc((B->rows_local+1)*sizeof(int));
        B->col_idx = (int*)malloc(B->nnz_local*sizeof(int));
        B->value = (double*)malloc(B->nnz_local*sizeof(double));
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

int check_sparse_AT_plus_A(CSR B, CSR A)
{
    int id = A.rank;
    if(A.rank == 0) printf("Begin Check AT+A\n");
    CSR C;
    check_sparse_AT_plus_A_kernel(&C, &A);
    if(B.rank        != C.rank)        { printf("Rank %2d Error in rank\n", id); exit(-1); }
    if(B.nprocs      != C.nprocs)      { printf("Rank %2d Error in nprocs\n", id); exit(-1); }
    if(B.rows        != C.rows)        { printf("Rank %2d Error in rows\n", id); exit(-1); }
    if(B.cols        != C.cols)        { printf("Rank %2d Error in cols\n", id); exit(-1); }
    if(B.nnz         != C.nnz)         { printf("Rank %2d Error in nnz\n", id); exit(-1); }
    if(B.rows_local  != C.rows_local)  { printf("Rank %2d Error in rows_local\n", id); exit(-1); }
    if(B.rows_offset != C.rows_offset) { printf("Rank %2d Error in rows_offset\n", id); exit(-1); }
    if(B.nnz_local   != C.nnz_local)   { printf("Rank %2d Error in nnz_local\n", id); exit(-1); }

    for(int i = 0; i <= B.rows_local; i ++)
    {
        if(B.row_ptr[i] != C.row_ptr[i]) {printf("Rank %2d Error in row_ptr[%d](%d,%d)\n", id, i, B.row_ptr[i], C.row_ptr[i]); exit(-1);}
    }
    int offset = B.row_ptr[0];
    for(int i = 0; i < B.rows_local; i ++)
        for(int j = B.row_ptr[i]-offset; j < B.row_ptr[i+1]-offset; j ++)
            if(B.col_idx[j] != C.col_idx[j]) {printf("Rank %2d Error in Row %d col_idx[%d](%d,%d)\n", id, i, j, B.col_idx[j], C.col_idx[j]); exit(-1);}
    for(int i = 0; i < B.rows_local; i ++)
        for(int j = B.row_ptr[i]-offset; j < B.row_ptr[i+1]-offset; j ++)
            if(fabs(B.value[j]-C.value[j]) > 1e-8) {printf("Rank %2d Error in Row %d value[%d](%lf,%lf)\n", id, i, j, B.value[j], C.value[j]); exit(-1);}
    MPI_Barrier(MPI_COMM_WORLD);
    if(A.rank == 0) printf("End Check AT+A\n");
    return 0;
}

int check_sparse_ATA(CSR B, CSR A)
{
    int id = A.rank;
    if(A.rank == 0) printf("Begin Check AT*A\n");
    CSR C;
    check_sparse_ATA_kernel(&C, &A);
    if(B.rank        != C.rank)        { printf("Rank %2d Error in rank\n", id); exit(-1); }
    if(B.nprocs      != C.nprocs)      { printf("Rank %2d Error in nprocs\n", id); exit(-1); }
    if(B.rows        != C.rows)        { printf("Rank %2d Error in rows\n", id); exit(-1); }
    if(B.cols        != C.cols)        { printf("Rank %2d Error in cols\n", id); exit(-1); }
    if(B.nnz         != C.nnz)         { printf("Rank %2d Error in nnz\n", id); exit(-1); }
    if(B.rows_local  != C.rows_local)  { printf("Rank %2d Error in rows_local\n", id); exit(-1); }
    if(B.rows_offset != C.rows_offset) { printf("Rank %2d Error in rows_offset\n", id); exit(-1); }
    if(B.nnz_local   != C.nnz_local)   { printf("Rank %2d Error in nnz_local\n", id); exit(-1); }

    for(int i = 0; i <= B.rows_local; i ++)
    {
        if(B.row_ptr[i] != C.row_ptr[i]) {printf("Rank %2d Error in row_ptr[%d](%d,%d)\n", id, i, B.row_ptr[i], C.row_ptr[i]); exit(-1);}
    }
    int offset = B.row_ptr[0];
    for(int i = 0; i < B.rows_local; i ++)
        for(int j = B.row_ptr[i]-offset; j < B.row_ptr[i+1]-offset; j ++)
            if(B.col_idx[j] != C.col_idx[j]) {printf("Rank %2d Error in Row %d col_idx[%d](%d,%d)\n", id, i, j, B.col_idx[j], C.col_idx[j]); exit(-1);}
    for(int i = 0; i < B.rows_local; i ++)
        for(int j = B.row_ptr[i]-offset; j < B.row_ptr[i+1]-offset; j ++)
            if(fabs(B.value[j]-C.value[j]) > 1e-8) {printf("Rank %2d Error in Row %d value[%d](%lf,%lf)\n", id, i, j, B.value[j], C.value[j]); exit(-1);}
    MPI_Barrier(MPI_COMM_WORLD);
    if(A.rank == 0) printf("End Check AT*A\n");
    return 0;
}


