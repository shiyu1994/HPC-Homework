#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mpi.h>
#include <string.h>
#include "func.h"
#include <iostream>

using std::cout;
using std::endl;

extern int output(int rows, int *row_ptr, int *col_idx, double *val);
extern void Qsort(int *a, double *b, int low, int high);

/* B = AT+A */
//int sparse_AT_plus_A_type0(CSR *B, CSR *A, bool first_time)
//{
//    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
/*    {
    }
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

        if(first_time)
        {
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
        if(first_time)
        {
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
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}*/


/* B = AT+A */
int sparse_AT_plus_A_type0(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {
    }
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

        if(first_time)
        {
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
        if(first_time)
        {
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
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}


/* B = AT+A */
int sparse_AT_plus_A_type1(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {
    }
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

        if(first_time)
        {
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
        if(first_time)
        {
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
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

/* B = AT*A */
int sparse_ATA_type0(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {
    }
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

        if(first_time)
        {
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
        if(first_time)
        {
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
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

/* B = AT*A */
int sparse_ATA_type1(CSR *B, CSR *A, bool first_time)
{
    if(first_time) /* The first time to use Matrix A, you can do some pre-process. */
    {
    }
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

        if(first_time)
        {
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
        if(first_time)
        {
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
        }
        MPI_Recv(B->row_ptr, B->rows_local+1, MPI_INT,    0, 4, MPI_COMM_WORLD, &status);
        MPI_Recv(B->col_idx, B->nnz_local,    MPI_INT,    0, 5, MPI_COMM_WORLD, &status);
        MPI_Recv(B->value,   B->nnz_local,    MPI_DOUBLE, 0, 6, MPI_COMM_WORLD, &status);
    }
    return 0;
}

