#include "sparse_ma.h"
#include <omp.h>
#include <algorithm>
#include <cassert>
//#include <hbwmalloc.h>

SparseMA::SparseMA(CSR *AA, CSR *BB, int num_threadss) {
    A = AA;
    B = BB;
    num_threads = num_threadss;
}

void SparseMA::Compute0(bool is_first_time) { 
    ComputeStandAlonePreprocess(is_first_time);
}

void SparseMA::Compute1(bool is_first_time) { 
    ComputeStandAlonePreprocess(is_first_time);
}


struct IdxValue {
    int idx;
    double value;
    IdxValue(int idxx, double valuee) {
        idx = idxx;
        value = value;
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

void SparseMA::ComputeStandAlonePreprocess(bool is_first_time) {
    if(is_first_time) {
    double t_start = omp_get_wtime();  

    col_to_row_map_size.resize(num_threads * (A->rows + 1), 0);
    col_to_row_map_cnt.resize(num_threads * (A->rows), 0);
   
    int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads) 
    for(int tid = 0; tid < num_threads; ++tid) {
        int start = chunk_size * tid;
        int end = std::min(start + chunk_size, A->rows);
        int *thread_col_to_row_map_size = col_to_row_map_size.data() + tid * (A->rows + 1);
        for(int i = start; i < end; ++i) {  
            int row_offset = A->row_ptr[i];
            int* row_idx = A->col_idx + row_offset;
            double* row_values = A->value + row_offset;
            int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i]; 
            for(int j = 0; j < row_nnz; ++j) {
                ++thread_col_to_row_map_size[row_idx[j] + 1];
            } 
        }
    } 

#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int *thread_col_to_row_map_size = col_to_row_map_size.data() + i * (A->rows + 1);
        for(int j = 0; j < A->rows; ++j) {
            thread_col_to_row_map_size[j + 1] += thread_col_to_row_map_size[j]; 
        }
    }

    for(int i = 1; i < num_threads; ++i) {
        col_to_row_map_size[i * (A->rows + 1) + A->rows] += col_to_row_map_size[(i - 1) * (A->rows + 1) + A->rows];
    }

#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 1; i < num_threads; ++i) {
        int offset = col_to_row_map_size[(i - 1) * (A->rows + 1) + A->rows];
#pragma omp simd
        for(int j = 0; j < A->rows; ++j) {
            col_to_row_map_size[i * (A->rows + 1) + j] += offset;
        }
    }

    int all_size = col_to_row_map_size[(num_threads - 1) * (A->rows + 1) + A->rows]; 
 
    col_to_row_map1.resize(all_size, 0);
    col_to_row_map2.resize(all_size, 0.0); 
 
#pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
        int *thread_col_to_row_map_cnt = col_to_row_map_cnt.data() + i * (A->rows);
        int *thread_col_to_row_map_size = col_to_row_map_size.data() + i * (A->rows + 1);
        for(int j = start; j < end; ++j) {
            int row_offset = A->row_ptr[j];
            int *row_idx = A->col_idx + row_offset;
            double *row_values = A->value + row_offset;
            int row_nnz = A->row_ptr[j + 1] - A->row_ptr[j];
            for(int k = 0; k < row_nnz; ++k) {
                int idx = row_idx[k];
                int col_to_row_offset = thread_col_to_row_map_size[idx];
                int &cnt = thread_col_to_row_map_cnt[idx];
                int index = col_to_row_offset + cnt;  
                col_to_row_map1[index] = j;
                col_to_row_map2[index] = row_values[k]; 
                ++cnt;
            }
        }
    } 

    double t_end = omp_get_wtime(); 

    t_start = omp_get_wtime(); 
    
    B->row_ptr = new int[A->rows + 1]; 
    vector<int> thread_offsets(num_threads + 1, 0);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int tid = 0; tid < num_threads; ++tid) {
        int start = tid * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
    for(int i = start; i < end; ++i) {  
        int row_offset = A->row_ptr[i];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i];
        B->row_ptr[i + 1] = row_nnz; 
        for(int j = 0; j < num_threads; ++j) {
            for(int k = 0; k < col_to_row_map_size[j * (A->rows + 1) + i + 1] - col_to_row_map_size[j * (A->rows + 1) + i]; ++k) {
                int idx_value = col_to_row_map1[col_to_row_map_size[j * (A->rows + 1) + i] + k];
                int pos = std::lower_bound(row_idx, row_idx + row_nnz, idx_value, comp) - row_idx;
                if(pos == row_nnz || row_idx[pos] != idx_value) { 
                    ++B->row_ptr[i + 1];
                }
            }
        }
        thread_offsets[tid + 1] += B->row_ptr[i + 1];
    }
    }

    for(int i = 0; i < num_threads; ++i) {
        thread_offsets[i + 1] += thread_offsets[i];
    }
    
    B->nnz = thread_offsets.back();
    B->rank = A->rank;
    B->nprocs = 1;
    B->rows = A->rows;
    B->cols = A->cols; 
    B->rows_local = A->rows;
    B->rows_offset = 0;
    B->nnz_local = B->nnz; 
    B->col_idx = new int[B->nnz];
    B->value = new double[B->nnz]; 

#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
        B->row_ptr[start] = thread_offsets[i];
        for(int j = start + 1; j < end; ++j) {
            B->row_ptr[j] += B->row_ptr[j - 1];
        }
    }
    B->row_ptr[A->rows] += B->row_ptr[A->rows - 1];
 
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < A->rows; ++i) { 
        int row_offset = A->row_ptr[i];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i]; 
        int offset = B->row_ptr[i];
#pragma omp simd
        for(int j = 0; j < row_nnz; ++j) {
            B->col_idx[offset + j] = row_idx[j]; 
            B->value[offset + j] = row_values[j]; 
        }
        int cnt = row_nnz;
        int size = B->row_ptr[i + 1] - B->row_ptr[i];
        for(int j = 0; j < num_threads; ++j) {
            int inner_offset = col_to_row_map_size[j * (A->rows + 1) + i];  
            for(int k = 0; k < col_to_row_map_size[j * (A->rows + 1) + i + 1] - col_to_row_map_size[j * (A->rows + 1) + i]; ++k) {
                int _idx = col_to_row_map1[inner_offset + k];
                double _value = col_to_row_map2[inner_offset + k]; 
                int pos = std::lower_bound(B->col_idx + offset, B->col_idx + offset + cnt, _idx, comp) - B->col_idx - offset; 
                if(pos == cnt || B->col_idx[offset + pos] != _idx) { 
#pragma omp simd
                    for(int tt = cnt; tt > pos; --tt) {
                        B->col_idx[offset + tt] = B->col_idx[offset + tt - 1];
                        B->value[offset + tt] = B->value[offset + tt - 1];
                    } 
                    B->col_idx[offset + pos] = _idx;
                    B->value[offset + pos] = _value;
                    ++cnt;
                } 
                else { 
                    B->value[offset + pos] += _value;  
                }
            }
        }
    }

    t_end = omp_get_wtime();  
    }
    else { 
        col_to_row_map_cnt.clear();
    col_to_row_map_cnt.resize(num_threads * (A->rows), 0);  
    int chunk_size = (A->rows + num_threads - 1) / num_threads;
    #pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = std::min(start + chunk_size, A->rows);
        int *thread_col_to_row_map_cnt = col_to_row_map_cnt.data() + i * (A->rows);
        int *thread_col_to_row_map_size = col_to_row_map_size.data() + i * (A->rows + 1);
        for(int j = start; j < end; ++j) {
            int row_offset = A->row_ptr[j];
            int *row_idx = A->col_idx + row_offset;
            double *row_values = A->value + row_offset;
            int row_nnz = A->row_ptr[j + 1] - A->row_ptr[j];
            for(int k = 0; k < row_nnz; ++k) {
                int idx = row_idx[k];
                int col_to_row_offset = thread_col_to_row_map_size[idx];
                int &cnt = thread_col_to_row_map_cnt[idx];
                int index = col_to_row_offset + cnt;  
                col_to_row_map1[index] = j;
                col_to_row_map2[index] = row_values[k]; 
                ++cnt;
            }
        }
    } 
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < A->rows; ++i) { 
        int row_offset = A->row_ptr[i];
        int* row_idx = A->col_idx + row_offset;
        double* row_values = A->value + row_offset;
        int row_nnz = A->row_ptr[i + 1] - A->row_ptr[i]; 
        int offset = B->row_ptr[i];
#pragma omp simd
        for(int j = 0; j < row_nnz; ++j) {
            B->col_idx[offset + j] = row_idx[j]; 
            B->value[offset + j] = row_values[j]; 
        }
        int cnt = row_nnz;
        int size = B->row_ptr[i + 1] - B->row_ptr[i];
        for(int j = 0; j < num_threads; ++j) {
            int inner_offset = col_to_row_map_size[j * (A->rows + 1) + i];  
            for(int k = 0; k < col_to_row_map_size[j * (A->rows + 1) + i + 1] - col_to_row_map_size[j * (A->rows + 1) + i]; ++k) {
                int _idx = col_to_row_map1[inner_offset + k];
                double _value = col_to_row_map2[inner_offset + k]; 
                int pos = std::lower_bound(B->col_idx + offset, B->col_idx + offset + cnt, _idx, comp) - B->col_idx - offset; 
                if(pos == cnt || B->col_idx[offset + pos] != _idx) { 
#pragma omp simd
                    for(int tt = cnt; tt > pos; --tt) {
                        B->col_idx[offset + tt] = B->col_idx[offset + tt - 1];
                        B->value[offset + tt] = B->value[offset + tt - 1];
                    } 
                    B->col_idx[offset + pos] = _idx;
                    B->value[offset + pos] = _value;
                    ++cnt;
                } 
                else { 
                    B->value[offset + pos] += _value;  
                }
            }
        }
    }

    }
}
