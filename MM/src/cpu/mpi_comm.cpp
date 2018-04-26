//
//  mpi_comm.cpp
//  MatrixMultiply
//
//  Created by Shi Yu on 2017/12/17.
//  Copyright © 2017年 Shi Yu. All rights reserved.
//

#include "mpi_comm.hpp"
#include <mpi.h>
#include <omp.h>
#include <iostream>

using std::cout;
using std::endl;

MatrixComm::MatrixComm(int dimm, int num_threadss) { dim = dimm; num_threads = num_threadss; }

//init MPI Communication
void MatrixComm::Init(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_machines);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Partition();
}

//distribute matrix to machines
void MatrixComm::SendMatrix(const dmatrix64 &matrix1, const dmatrix64 &matrix2) {
    //rank 0 machine distributes the matrix to other machines
    if(num_machines % 2 == 0) {
        if(rank == 0) {
            //send matrix1
            for(int i = 1; i < num_machines; ++i) {
                for(int row = local_row_starts[i]; row < local_row_ends[i]; ++row) {
                    MPI_Send(matrix1[row].data(), dim, MPI_DOUBLE, i, row * 2, MPI_COMM_WORLD);
                }
            }
            //send matrix2
            for(int i = 1; i < num_machines; ++i) {
                for(int row = 0; row < dim; ++row) {
                    int row_len = local_col_ends[i] - local_col_starts[i];
                    MPI_Send(matrix2[row].data() + local_col_starts[i],
                             row_len, MPI_DOUBLE, i, row * 2 + 1, MPI_COMM_WORLD);  
                }
            }
        }
    }
    else {
        //TODO
    }
}

//receive matrix data from rank 0 machine
void MatrixComm::ReceiveMatrix(dmatrix64 &matrix1, dmatrix64 &matrix2) {
    if(num_machines % 2 == 0) {
        if(rank != 0) {
            int row_len = local_col_end - local_col_start;
            MPI_Status status;
            //receive matrix1
            for(int i = 0; i < local_row_end - local_row_start; ++i) {
                MPI_Recv(matrix1[i].data(), dim, MPI_DOUBLE, 0, (i + local_row_start) * 2, MPI_COMM_WORLD, &status);
            }
            //receivee matrix2
            for(int i = 0; i < dim; ++i) {
                MPI_Recv(matrix2[i].data(), row_len, MPI_DOUBLE, 0, i * 2 + 1, MPI_COMM_WORLD, &status);
            }
        }
    }
    else {
        //TODO
    }
}

//gather results from machines
void MatrixComm::GatherMatrix(dmatrix64 &matrix) {
    if(num_machines % 2 == 0) {
        if(rank != 0) {
            int row_len = local_col_end - local_col_start;
            for(int row = 0; row < local_row_end - local_row_start; ++row) {
                MPI_Send(matrix[row].data(), row_len, MPI_DOUBLE, 0,
                         2 * (row + local_row_start) + rank % 2, MPI_COMM_WORLD);
            }
        }
        else {
            MPI_Status status;
            for(int i = 1; i < num_machines; ++i) {
                int row_len = local_col_ends[i] - local_col_starts[i];
                for(int j = local_row_starts[i]; j < local_row_ends[i]; ++j) {
                    MPI_Recv(matrix[j].data() + local_col_starts[i], row_len,
                             MPI_DOUBLE, i, j * 2 + i % 2, MPI_COMM_WORLD, &status);        
                }
            }
        }
    }
    else {
        //TODO
    }
}

//calculate boundaries of machines
void MatrixComm::Partition() {
    if(num_machines % 2 == 0) {
        //divide right matrix horizontally into two parts
        if(rank % 2 == 0) {
            local_col_start = 0;
            local_col_end = dim / 2;
        }
        else {
            local_col_start = dim / 2;
            local_col_end = dim;
        }
        //divide the left matrix veritically evenly to all machines
        int half_num_machines = num_machines / 2;
        int rows_per_machine = (dim + half_num_machines - 1) / half_num_machines;
        local_row_start = rows_per_machine * (rank / 2);
        local_row_end = std::min(local_row_start + rows_per_machine, dim);
        if(rank == 0) {
            local_row_starts.resize(num_machines);
            local_row_ends.resize(num_machines);
            local_col_starts.resize(num_machines);
            local_col_ends.resize(num_machines);
            for(int j = 0; j < num_machines; ++j) {
                if(j % 2 == 0) {
                    local_col_starts[j] = 0;
                    local_col_ends[j] = dim / 2;
                }
                else {
                    local_col_starts[j] = dim / 2;
                    local_col_ends[j] = dim;
                }
                local_row_starts[j] = half_num_machines * (j / 2);
                local_row_ends[j] = std::min(local_row_starts[j] + rows_per_machine, dim);
            }
        }
    }
    else {
        //TODO
    }
}

void MatrixComm::InitMatrix(dmatrix64 &matrix1, dmatrix64 &matrix2, dmatrix64 &matrix3) {
    srand(static_cast<int>(std::time(nullptr)));
    if(rank == 0) {
        //rank 0 machine prepares matrix data
        matrix1.resize(dim);
        matrix2.resize(dim);
        matrix3.resize(dim);
        //generate randon data
#pragma omp parallel for schedule(static) num_threads(num_threads)  
        for(int i = 0; i < dim; ++i) {
            matrix1[i].resize(dim, 0.0);
            matrix2[i].resize(dim, 0.0);
            matrix3[i].resize(dim, 0.0);
            for(int j = 0; j < dim; ++j) {
                matrix1[i][j] = std::rand() * 1.0 / RAND_MAX;
                matrix2[i][j] = std::rand() * 1.0 / RAND_MAX;
            }
        }
    }
    else {
        //other machines prepare space to receive matrix
        int num_rows = local_row_end - local_row_start;
        int num_cols = local_col_end - local_col_start;
        matrix1.resize(num_rows);
        matrix2.resize(dim);
        matrix3.resize(num_rows);
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < num_rows; ++i) {
            matrix1[i].resize(dim, 0.0);
            matrix3[i].resize(num_cols, 0.0);
        }
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < dim; ++i) {
            matrix2[i].resize(num_cols, 0.0);
        }
    }
}

void MatrixComm::Finalize() {
    MPI_Finalize(); 
}

void MatrixComm::MPICompute(function<void (int, int, int)> compute_kernel) {
    compute_kernel(local_row_end - local_row_start, dim, local_col_end - local_col_start);
}

void MatrixComm::MPICheck(function<void ()> check_kernel) {
    if(rank == 0) {
        check_kernel(); 
    }
}
