#include "sparse_ma.h"
#include <omp.h>
#include <algorithm>
#include <cassert>
#include <mpi.h>

SparseMA::SparseMA(CSR *AA, CSR *BB, int num_threadss) {
    A = AA;
    B = BB;
    num_threads = num_threadss;
}


void SparseMA::GatherData(bool is_first_time) { 
    if(is_first_time) { 
        A_nprocs = A->nprocs;  
        A_nnz = A->nnz; 
        A_row_ptr.resize(A->rows + 1); 
        A_col_idx.resize(A_nnz);
        A_value.resize(A_nnz); 
        A_nnz_offsets.resize(A_nprocs + 1, 0);
        A_rank = A->rank; 
        if(A_rank == 0) {
            memcpy(A_row_ptr.data(), A->row_ptr, (A->rows_local+1)*sizeof(int));
            memcpy(A_col_idx.data(), A->col_idx, A->nnz_local*sizeof(int));
            memcpy(A_value.data(),   A->value,   A->nnz_local*sizeof(double));
            int rows_offset = A->rows_local;
            int nnz_offset = A->nnz_local;  
            MPI_Status status; 
            for(int i = 1; i < A_nprocs; i ++) {
                int rows_i;
                int nnz_i; 
                MPI_Recv(&rows_i, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&nnz_i,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(A_row_ptr.data() + rows_offset, rows_i+1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(A_col_idx.data() + nnz_offset,  nnz_i,    MPI_INT, i, 3, MPI_COMM_WORLD, &status);
                MPI_Recv(A_value.data()   + nnz_offset,  nnz_i, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
                A_nnz_offsets[i] = nnz_offset;
                rows_offset += rows_i;
                nnz_offset += nnz_i; 
            } 
            A_nnz_offsets[A_nprocs] = nnz_offset; 
            vector<MPI_Request> rsend1(A_nprocs);
            vector<MPI_Request> rsend2(A_nprocs);
            vector<MPI_Request> rsend3(A_nprocs);
            for(int i = 1; i < A_nprocs; ++i) {
                MPI_Status status;
                MPI_Isend(A_row_ptr.data(), A->rows + 1, MPI_INT, i, 5, MPI_COMM_WORLD, &rsend1[i]);
                MPI_Isend(A_col_idx.data(), A_nnz, MPI_INT, i, 6, MPI_COMM_WORLD, &rsend2[i]);
                MPI_Isend(A_value.data(), A_nnz, MPI_DOUBLE, i, 7, MPI_COMM_WORLD, &rsend3[i]);
            }
            for(int i = 1; i < A_nprocs; ++i) {
                MPI_Wait(&rsend1[i], &status);
                MPI_Wait(&rsend2[i], &status);
                MPI_Wait(&rsend3[i], &status);
            }
        }
        else {
            MPI_Status status; 
            MPI_Send(&A->rows_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&A->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(A->row_ptr, A->rows_local+1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(A->col_idx, A->nnz_local,    MPI_INT, 0, 3, MPI_COMM_WORLD);
            MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);  
            MPI_Request rrecv1, rrecv2, rrecv3;
            MPI_Irecv(A_row_ptr.data(), A->rows + 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &rrecv1);
            MPI_Irecv(A_col_idx.data(), A_nnz, MPI_INT, 0, 6, MPI_COMM_WORLD, &rrecv2);
            MPI_Irecv(A_value.data(), A_nnz, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD, &rrecv3);    
            MPI_Wait(&rrecv1, &status);
            MPI_Wait(&rrecv2, &status);
            MPI_Wait(&rrecv3, &status);
        }
    }
    else {
        if(A_rank == 0) {
            vector<MPI_Request> rsend(A_nprocs);
            MPI_Status status; 
            memcpy(A_value.data(),   A->value,   A->nnz_local*sizeof(double));
            for(int i = 1; i < A_nprocs; i ++) {   
                MPI_Recv(A_value.data() + A_nnz_offsets[i],  A_nnz_offsets[i + 1] - A_nnz_offsets[i], MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &status);       
            } 
            for(int i = 1; i < A_nprocs; ++i) { 
                MPI_Isend(A_value.data(), A_nnz, MPI_DOUBLE, i, 7, MPI_COMM_WORLD, &rsend[i]);
            }
            for(int i = 1; i < A_nprocs; ++i) {
                MPI_Wait(&rsend[i], &status);
            }
        }
        else {
            MPI_Status status;
            MPI_Request rrecv; 
            MPI_Send(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);             
            MPI_Irecv(A_value.data(), A_nnz, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD, &rrecv);    
            MPI_Wait(&rrecv, &status);
        }
    }   
}


int SparseMA::GatherNNZ() {
    if(A_rank == 0) {
          int all_nnz = B->nnz_local;
          row_nnz_offsets.resize(A_nprocs + 1, 0);
          row_nnz_offsets[1] = B->nnz_local;
            for(int i = 1; i < A_nprocs; i ++) { 
                int nnz_i;
                MPI_Status status; 
                MPI_Recv(&nnz_i,  1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
                row_nnz_offsets[i + 1] = nnz_i;
                all_nnz += nnz_i; 
            } 
          
            for(int i = 0; i < A_nprocs; ++i) {
                row_nnz_offsets[i + 1] += row_nnz_offsets[i];
            }
            for(int i = 1; i < A_nprocs; ++i) {
                MPI_Status status;
                MPI_Send(&all_nnz, 1, MPI_INT, i, 2, MPI_COMM_WORLD);
                MPI_Send(row_nnz_offsets.data(), A_nprocs + 1, MPI_INT, i, 3, MPI_COMM_WORLD);
            }
            return all_nnz;
        }
        else {
            int all_nnz = 0;
            row_nnz_offsets.resize(A_nprocs + 1, 0);
            MPI_Status status; 
            MPI_Send(&B->nnz_local,  1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Recv(&all_nnz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);    
            MPI_Recv(row_nnz_offsets.data(), A_nprocs + 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);    
            return all_nnz;
        }

}

void SparseMA::Compute0(bool is_first_time) {
    ComputeWithPreprocess(is_first_time);
}

void SparseMA::Compute1(bool is_first_time) {
    ComputeWithPreprocess(is_first_time);
}


struct IdxValue {
    int idx;
    double value; 
    IdxValue(int idxx, double valuee) {
        idx = idxx;
        value = valuee;
    }
};

static bool comp(int a, int b) { return a < b; }

struct IdxValueAddr {
    int idx;
    double value; 
    double *addr;
    double *addr_t;
    IdxValueAddr(int idxx, double valuee, double *addrr, bool from_trans) {
        idx = idxx;
        value = valuee;
        if(from_trans) {
            addr_t = addrr;
            addr = nullptr;
        }
        else {
            addr = addrr;
            addr_t = nullptr;
        }
    }
};

void SparseMA::ComputeWithPreprocess(bool is_first_time) { 
    double t_start = omp_get_wtime();
    GatherData(is_first_time);
    double t_end = omp_get_wtime(); 
    MPI_Barrier(MPI_COMM_WORLD); 
    if(is_first_time) {
    double t_start = omp_get_wtime();
    vector<vector<vector<IdxValueAddr*>>> col_to_row_map(num_threads); 
     
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        col_to_row_map[i].resize(A->rows_local); 
    }
   
#pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 0; i < A->rows; ++i) { 
        int tid = omp_get_thread_num();
        int row_offset = A_row_ptr[i];
        int* row_idx = A_col_idx.data() + row_offset;
        double* row_values = A_value.data() + row_offset;
        int row_nnz = A_row_ptr[i + 1] - A_row_ptr[i];
        for(int j = 0; j < row_nnz; ++j) {
            assert(row_offset + j < A->nnz);
            int row_idx_j = row_idx[j];
            if(row_idx_j >= A->rows_offset && row_idx_j < A->rows_offset + A->rows_local) {
                col_to_row_map[tid][row_idx_j - A->rows_offset].push_back(new IdxValueAddr(i, row_values[j], A_value.data() + j + A_row_ptr[i], true)); 
            }
        }
    } 

    double t_end = omp_get_wtime(); 
    
    t_start = omp_get_wtime();
    vector<vector<IdxValueAddr*>> all_col_to_row_map(A->rows_local);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < A->rows_local; ++i) {
        int tid = omp_get_thread_num();
        int row_offset = A->row_ptr[i] - A->row_ptr[0];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i];
        all_col_to_row_map[i].resize(row_nnz, nullptr);
        for(int j = 0; j < row_nnz; ++j) {
            all_col_to_row_map[i][j] = new IdxValueAddr(row_idx[j], row_values[j], A_value.data() + j + A_row_ptr[i + A->rows_offset], false);   
        }  
        for(int j = 0; j < num_threads; ++j) {
            for(int k = 0; k < col_to_row_map[j][i].size(); ++k) {
                IdxValueAddr* idx_value = col_to_row_map[j][i][k];
                int pos = std::lower_bound(row_idx, row_idx + row_nnz, idx_value->idx, comp) - row_idx;
                if(pos == row_nnz || all_col_to_row_map[i][pos]->idx != idx_value->idx) {
                    all_col_to_row_map[i].push_back(idx_value);
                } 
                else { 
                    all_col_to_row_map[i][pos]->value += idx_value->value;
                    all_col_to_row_map[i][pos]->addr_t = idx_value->addr_t;
                    delete idx_value;
                }
            }
        }
    }
    t_end = omp_get_wtime(); 

    col_to_row_map.clear();
    col_to_row_map.shrink_to_fit(); 
    
    t_start = omp_get_wtime();
    vector<int> thread_copy_offsets(num_threads + 1, 0);
    int chunk_size = (A->rows_local + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows_local);
        for(int j = start; j < end; ++j) {
            thread_copy_offsets[i + 1] += all_col_to_row_map[j].size();
        }
    }

    for(int i = 0; i < num_threads; ++i) {
        thread_copy_offsets[i + 1] += thread_copy_offsets[i];
    } 

    int nnz = thread_copy_offsets.back();

    B->rank = A->rank;
    B->nprocs = A->nprocs;
    B->rows = A->rows;
    B->cols = A->cols;
    B->nnz_local = nnz;
    B->nnz = GatherNNZ();

    B->col_idx = new int[nnz];
    B->value = new double[nnz];
    B->row_ptr = new int[A->rows_local + 1]; 
    B->row_ptr[0] = row_nnz_offsets[A_rank];
    B->rows_local = A->rows_local;
    B->rows_offset = A->rows_offset;

    addrs.resize(nnz, nullptr);
    addr_ts.resize(nnz, nullptr);

    t_end = omp_get_wtime();

    t_start = omp_get_wtime(); 
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows_local);
        int cur_pos = thread_copy_offsets[i];
        for(int j = start; j < end; ++j) {
            std::sort(all_col_to_row_map[j].begin(), all_col_to_row_map[j].end(), [] (IdxValueAddr* a, IdxValueAddr* b) -> bool { return a->idx < b->idx;});
            for(int k = 0; k < all_col_to_row_map[j].size(); ++k) { 
                B->col_idx[cur_pos] = all_col_to_row_map[j][k]->idx;
                B->value[cur_pos] = all_col_to_row_map[j][k]->value;
                addrs[cur_pos] = all_col_to_row_map[j][k]->addr;
                addr_ts[cur_pos] = all_col_to_row_map[j][k]->addr_t;
                ++cur_pos;
            } 
            B->row_ptr[j + 1] = cur_pos + B->row_ptr[0];
        } 
    }  
    t_end = omp_get_wtime(); 
    }
    else { 
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < A->rows_local; ++i) {
            int offset = B->row_ptr[i] - B->row_ptr[0];
            int nnz = B->row_ptr[i + 1] - B->row_ptr[i];
            for(int j = 0; j < nnz; ++j) {
                assert(offset + j < B->nnz_local);
                if(addrs[offset + j] != nullptr && addr_ts[offset + j] != nullptr) {
                    B->value[offset + j] = *addrs[offset + j] + *addr_ts[offset + j];
                }
                else if(addrs[offset + j] == nullptr) {
                    assert(addr_ts[offset + j] != nullptr);
                    B->value[offset + j] = *addr_ts[offset + j];
                }
                else {
                    assert(addrs[offset] + j != nullptr);
                    B->value[offset + j] = *addrs[offset + j];
                }
            }
        } 
    }
}


