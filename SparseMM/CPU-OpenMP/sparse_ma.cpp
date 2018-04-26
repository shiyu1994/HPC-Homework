#include "sparse_ma.h"
#include <omp.h>
#include <algorithm>
#include <cassert>

SparseMA::SparseMA(CSR *AA, CSR *BB, int num_threadss) {
    A = AA;
    B = BB;
    num_threads = num_threadss;
}

void SparseMA::PartitionTasks() {
    /*MPI_Request send_request1, send_request2, recv_request1, recv_request2;
    MPI_Status status;
    for(int i = 0; i < A->nprocs; ++i) {
        MPI_Isend(&(A->rows_offset), 1, MPI_INT, i, A->rank * 2, MPI_COMM_WORLD, &send_request1);
        MPI_Isend(&(A->rows_local), 1, MPI_INT, i, A->rank * 2 + 1, MPI_COMM_WORLD, &send_request2);
        MPI_Irecv(&rows_offset[i], 1, MPI_INT, i, i * 2, MPI_COMM_WORLD, &recv_request1);
        MPI_Irecv(&rows_local[i], 1, MPI_INT, i, i * 2 + 1, MPI_COMM_WORLD, &recv_request2);
    }
    MPI_Wait(&send_request1, &status);
    MPI_Wait(&send_request1, &status);
    MPI_Wait(&recv_request1, &status);
    MPI_Wait(&recv_request1, &status);*/
}

void SparseMA::GatherData() {
    /*int rank = A->rank;
    if(rank == 0) {
        rows.resize(A->rows_local);
        memcpy(rows.data(), A->row_ptr, sizeof(int) * A->rows_local);
        for(int i = 1; i < A->nprocs; ++i) {
            
        }
    }*/
}

void SparseMA::Compute(bool is_first_time) {
    //PartitionTasks();
    //GatherData();
    ComputeStandAlonePreprocess(is_first_time);
}

struct IdxValue {
    int idx;
    double value;
    //SourceAddr *source_addr;
    //double *addr;
    IdxValue(int idxx, double valuee) {//, double *addrr) {
        idx = idxx;
        value = valuee;
        //addr = addrr;
    }
};

static bool comp(int a, int b) { return a < b; }

void SparseMA::ComputeStandAlone() {
    double t_start = omp_get_wtime();
    vector<vector<vector<IdxValue*>>> col_to_row_map(num_threads); 
     
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        col_to_row_map[i].resize(A->rows); 
    }
   
#pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 0; i < A->rows; ++i) { 
        int tid = omp_get_thread_num();
        int row_offset = A->row_ptr[i];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i];
        for(int j = 0; j < row_nnz; ++j) {
            col_to_row_map[tid][row_idx[j]].push_back(new IdxValue(i, row_values[j])); 
        }
    } 

    double t_end = omp_get_wtime();
    //cout << "step 1 time: " << (t_end - t_start) << endl;
    
    t_start = omp_get_wtime();
    vector<vector<IdxValue*>> all_col_to_row_map(A->rows);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < A->rows; ++i) {
        int tid = omp_get_thread_num();
        int row_offset = A->row_ptr[i];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i];
        all_col_to_row_map[i].resize(row_nnz, nullptr);
        for(int j = 0; j < row_nnz; ++j) {
            all_col_to_row_map[i][j] = new IdxValue(row_idx[j], row_values[j]);   
        }  
        for(int j = 0; j < num_threads; ++j) {
            for(int k = 0; k < col_to_row_map[j][i].size(); ++k) {
                IdxValue* idx_value = col_to_row_map[j][i][k];
                int pos = std::lower_bound(row_idx, row_idx + row_nnz, idx_value->idx, comp) - row_idx;
                if(pos == row_nnz || all_col_to_row_map[i][pos]->idx != idx_value->idx) {
                    all_col_to_row_map[i].push_back(idx_value);
                } 
                else { 
                    all_col_to_row_map[i][pos]->value += idx_value->value;
                    delete idx_value;
                }
            }
        }
    }
    t_end = omp_get_wtime();
    //cout << "step 2 time: " << (t_end - t_start) << endl;

    col_to_row_map.clear();
    col_to_row_map.shrink_to_fit(); 
    
    t_start = omp_get_wtime();
    vector<int> thread_copy_offsets(num_threads + 1, 0);
    int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
        for(int j = start; j < end; ++j) {
            thread_copy_offsets[i + 1] += all_col_to_row_map[j].size();
        }
    }

    for(int i = 0; i < num_threads; ++i) {
        thread_copy_offsets[i + 1] += thread_copy_offsets[i];
    }

    int nnz = thread_copy_offsets.back();
    vector<int> out_idx(nnz);
    vector<double> out_value(nnz);
    vector<int> out_row_ptr(A->rows + 1, 0); 

    t_end = omp_get_wtime();
 //cout << "step 3 time: " << (t_end - t_start) << endl;

    t_start = omp_get_wtime();
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
        int cur_pos = thread_copy_offsets[i];
        for(int j = start; j < end; ++j) {
            std::sort(all_col_to_row_map[j].begin(), all_col_to_row_map[j].end(), [] (IdxValue* a, IdxValue* b) -> bool { return a->idx < b->idx;});
            for(int k = 0; k < all_col_to_row_map[j].size(); ++k) { 
                out_idx[cur_pos] = all_col_to_row_map[j][k]->idx;
                out_value[cur_pos] = all_col_to_row_map[j][k]->value;
                ++cur_pos;
            } 
            out_row_ptr[j + 1] = cur_pos;
        } 
    }  
    t_end = omp_get_wtime();
 //cout << "step 4 time: " << (t_end - t_start) << endl;

    B->rank = A->rank;
    B->nprocs = 1;
    B->rows = A->rows;
    B->cols = A->cols;
    B->nnz = nnz;
    B->rows_local = A->rows;
    B->rows_offset = 0;
    B->nnz_local = nnz;
    B->row_ptr = new int[A->rows + 1];
    B->col_idx = new int[nnz];
    B->value = new double[nnz];
    memcpy(B->row_ptr, out_row_ptr.data(), sizeof(int) * (A->rows + 1));
    memcpy(B->col_idx, out_idx.data(), sizeof(int) * nnz);
    memcpy(B->value, out_value.data(), sizeof(double) * nnz);

}


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

void SparseMA::ComputeStandAlonePreprocess(bool is_first_time) {
    if(is_first_time) {
    double t_start = omp_get_wtime();
    vector<vector<vector<IdxValueAddr*>>> col_to_row_map(num_threads); 
     
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        col_to_row_map[i].resize(A->rows); 
    }
   
#pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 0; i < A->rows; ++i) { 
        int tid = omp_get_thread_num();
        int row_offset = A->row_ptr[i];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i];
        for(int j = 0; j < row_nnz; ++j) {
            col_to_row_map[tid][row_idx[j]].push_back(new IdxValueAddr(i, row_values[j], row_values + j, true)); 
        }
    } 

    double t_end = omp_get_wtime();
    //cout << "step 1 time: " << (t_end - t_start) << endl;
    
    t_start = omp_get_wtime();
    vector<vector<IdxValueAddr*>> all_col_to_row_map(A->rows);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < A->rows; ++i) {
        int tid = omp_get_thread_num();
        int row_offset = A->row_ptr[i];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i];
        all_col_to_row_map[i].resize(row_nnz, nullptr);
        for(int j = 0; j < row_nnz; ++j) {
            all_col_to_row_map[i][j] = new IdxValueAddr(row_idx[j], row_values[j], row_values + j, false);   
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
    //cout << "step 2 time: " << (t_end - t_start) << endl;

    col_to_row_map.clear();
    col_to_row_map.shrink_to_fit(); 
    
    t_start = omp_get_wtime();
    vector<int> thread_copy_offsets(num_threads + 1, 0);
    int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
        for(int j = start; j < end; ++j) {
            thread_copy_offsets[i + 1] += all_col_to_row_map[j].size();
        }
    }

    for(int i = 0; i < num_threads; ++i) {
        thread_copy_offsets[i + 1] += thread_copy_offsets[i];
    }

    int nnz = thread_copy_offsets.back();
    //vector<int> out_idx(nnz);
    //vector<double> out_value(nnz);
    //vector<int> out_row_ptr(A->rows + 1, 0); 
    col_idx.resize(nnz);
    values.resize(nnz);
    rows.resize(A->rows + 1, 0);
    addrs.resize(nnz, nullptr);
    addr_ts.resize(nnz, nullptr);

    t_end = omp_get_wtime();
 //cout << "step 3 time: " << (t_end - t_start) << endl;

    t_start = omp_get_wtime();
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
        int cur_pos = thread_copy_offsets[i];
        for(int j = start; j < end; ++j) {
            std::sort(all_col_to_row_map[j].begin(), all_col_to_row_map[j].end(), [] (IdxValueAddr* a, IdxValueAddr* b) -> bool { return a->idx < b->idx;});
            for(int k = 0; k < all_col_to_row_map[j].size(); ++k) { 
                col_idx[cur_pos] = all_col_to_row_map[j][k]->idx;
                values[cur_pos] = all_col_to_row_map[j][k]->value;
                addrs[cur_pos] = all_col_to_row_map[j][k]->addr;
                addr_ts[cur_pos] = all_col_to_row_map[j][k]->addr_t;
                ++cur_pos;
            } 
            rows[j + 1] = cur_pos;
        } 
    }  
    t_end = omp_get_wtime();
 //cout << "step 4 time: " << (t_end - t_start) << endl;

    B->rank = A->rank;
    B->nprocs = 1;
    B->rows = A->rows;
    B->cols = A->cols;
    B->nnz = nnz;
    B->rows_local = A->rows;
    B->rows_offset = 0;
    B->nnz_local = nnz;
    B->row_ptr = new int[A->rows + 1];
    B->col_idx = new int[nnz];
    B->value = new double[nnz];
    memcpy(B->row_ptr, rows.data(), sizeof(int) * (A->rows + 1));
    memcpy(B->col_idx, col_idx.data(), sizeof(int) * nnz);
    memcpy(B->value, values.data(), sizeof(double) * nnz);
    }
    else {
        //cout << "begin comp" << endl;
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < A->rows; ++i) {
            //cout << "here1" << endl;
            int offset = rows[i];
            int nnz = rows[i + 1] - rows[i];
            //cout << "here" << endl;
            for(int j = 0; j < nnz; ++j) {
                //cout << j << endl;
                assert(offset + j < B->nnz);
                if(addrs[offset + j] != nullptr && addr_ts[offset + j] != nullptr) {
                    values[offset + j] = *addrs[offset + j] + *addr_ts[offset + j];
                }
                else if(addrs[offset + j] == nullptr) {
                    assert(addr_ts[offset + j] != nullptr);
                    values[offset + j] = *addr_ts[offset + j];
                }
                else {
                    assert(addrs[offset] + j != nullptr);
                    values[offset + j] = *addrs[offset + j];
                }
            }
        }
    //cout << "finish comp" << endl;
        B->rank = A->rank;
        B->nprocs = 1;
        B->rows = A->rows;
        B->cols = A->cols;
        B->nnz = values.size();
        B->rows_local = A->rows;
        B->rows_offset = 0;
        B->nnz_local = values.size();

        memcpy(B->value, values.data(), sizeof(double) * B->nnz);

    }
}
