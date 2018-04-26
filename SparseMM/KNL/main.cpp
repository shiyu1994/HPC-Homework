#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mkl.h>
#include <mpi.h>

#include "func.h"

#define TIME(a,b)     (1.0*((b).tv_sec-(a).tv_sec)+0.000001*((b).tv_usec-(a).tv_usec))

int load_csr_distribute(CSR *A, char *filename)
{
    FILE *fin = fopen(filename, "rb");
    int rows, cols, nnz, rank, nprocs, rows_local, rows_offset, nnz_local;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fread(&rows, 1, sizeof(int), fin);
    fread(&cols, 1, sizeof(int), fin);
    fread(&nnz,  1, sizeof(int), fin);
    rows_local  = rows/nprocs        + (rank < (rows%nprocs) ? 1 : 0);
    rows_offset = rank*(rows/nprocs) + (rank < (rows%nprocs) ? rank : (rows%nprocs));
    A->rank        = rank;
    A->nprocs      = nprocs;
    A->rows        = rows;
    A->cols        = cols;
    A->nnz         = nnz;
    A->rows_local  = rows_local;
    A->rows_offset = rows_offset;

    A->row_ptr = (int*)malloc((rows_local+1)*sizeof(int));
    fseek(fin, (3+rows_offset)*sizeof(int), SEEK_SET);
    fread(A->row_ptr, sizeof(int), rows_local+1, fin);
    A->nnz_local = nnz_local = A->row_ptr[rows_local] - A->row_ptr[0];

    A->col_idx = (int*)malloc(nnz_local*sizeof(int));
    A->value   = (double*)malloc(nnz_local*sizeof(double));

    fseek(fin, (3+rows+1)*sizeof(int) + A->row_ptr[0]*sizeof(int), SEEK_SET);
    fread(A->col_idx, sizeof(int), nnz_local, fin);

    fseek(fin, (3+rows+1+nnz)*sizeof(int) + A->row_ptr[0]*sizeof(double), SEEK_SET);
    fread(A->value, sizeof(double), nnz_local, fin);
    
    fclose(fin);
    return 0; 
}

int destabilization(double *val, int n)
{
#define RAND (seed++/RAND_MAX)
    int i;
#pragma omp parallel
    {
        double seed = rand();
#pragma omp for
        for(i = 0; i < n; i ++)
        {
            val[i] = RAND*1e-8;
        }
    }
}

int destroy_CSR(CSR *A)
{
    A->rank = 0;
    A->nprocs = 0;
    A->rows = A->cols = 0;
    A->nnz = 0;
    A->rows_local = 0;
    A->rows_offset = 0;
    A->nnz_local = 0;
    free(A->row_ptr);
    free(A->col_idx);
    free(A->value);
    return 0;
}

int main(int argc, char **argv)
{
    CSR A;
    MPI_Init(&argc, &argv);
    load_csr_distribute(&A, argv[1]);

    int Warm_Steps[4] = {1,1,1,1};
    int Test_Steps[4] = {3,300,3,300};
    double time[4] = {0,0,0,0};
    int (*test_func[4])(CSR*, CSR*, bool) = {sparse_AT_plus_A_type0, sparse_AT_plus_A_type1, sparse_ATA_type0, sparse_ATA_type1};
    int (*check_func[4])(CSR, CSR) = {check_sparse_AT_plus_A, check_sparse_AT_plus_A, check_sparse_ATA, check_sparse_ATA};
    
    if(A.rank == 0) printf("Begin to warmup\n");
    for(int i = 0; i < 4; i ++)
    {
        CSR B;
        for(int j = 0; j < Warm_Steps[i]; j ++)
            test_func[i](&B, &A, (j==0));
        destroy_CSR(&B);
    }

    if(A.rank == 0) printf("Begin to test\n");
    for(int i = 0; i < 4; i ++)
    {
        struct timeval t1, t2;
        CSR B;
        for(int j = 0; j < Test_Steps[i]; j ++)
        {
            destabilization(A.value, A.nnz_local);
            MPI_Barrier(MPI_COMM_WORLD); gettimeofday(&t1, NULL);
            test_func[i](&B, &A, (j==0));
            MPI_Barrier(MPI_COMM_WORLD); gettimeofday(&t2, NULL);
            time[i] += TIME(t1,t2);
            if(A.rank == 0) printf("Step %3d %3d Time %.6lf\n", i, j, TIME(t1,t2)); //You can delete this line
        }
        check_func[i](B, A);
        destroy_CSR(&B);
    }

    if(A.rank == 0) printf("%s,%.6lf,%.6lf,%.6lf,%.6lf\n", argv[1], time[0], time[1], time[2], time[3]); //This is an important output

    destroy_CSR(&A);
    MPI_Finalize();
    return 0;
}
