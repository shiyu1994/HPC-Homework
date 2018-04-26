#include "sparse_mm.h"
#include <algorithm>
#include <set>
#include <map>
#include <omp.h>
#include <cassert>

using std::set;
using std::map;

SparseMM::SparseMM(CSR *AA, CSR *BB, int num_threadss) {
    A = AA;
    B = BB;
    num_threads = num_threadss;
}

void SparseMM::Compute0(bool is_first_time) {
    Compute1(is_first_time);
}

void SparseMM::Compute1(bool is_first_time) {
    if(is_first_time) { 
        int all_contri = 0;
        vector<int> thread_row_cnts(num_threads * A->rows, 0);
        vector<int> thread_row_contri_cnts(num_threads * A->rows, 0);
        vector<int> thread_row_contri_cnts_cnts(num_threads * A->rows, 0);
        vector<int> thread_row_contri_offsets(A->rows * (num_threads + 1), 0);
        int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads) reduction(+:all_contri)
        for(int tid = 0; tid < num_threads; ++tid) {
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            int* local_row_cnts = thread_row_cnts.data() + tid * A->rows;
            int* local_row_contri_cnts = thread_row_contri_cnts.data() + tid * A->rows;
            for(int i = start; i < end; ++i) {
                int offset = A->row_ptr[i];
                int nnz = A->row_ptr[i + 1] - offset;
                int* col_idx = A->col_idx + offset;
                for(int j = 0; j < nnz; ++j) {
                    int idx = col_idx[j];
                    local_row_cnts[idx] += nnz;
                    local_row_contri_cnts[idx] += 1;
                    ++all_contri;
                }  
            }
        }
        
        vector<int> thread_offsets(num_threads + 1, 0);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int i = start; i < end; ++i) {
                int* local_thread_row_contri_offsets = thread_row_contri_offsets.data() + i * (num_threads + 1);
                for(int t = 0; t < num_threads; ++t) {
                    local_thread_row_contri_offsets[t + 1] = local_thread_row_contri_offsets[t] + thread_row_contri_cnts[t * A->rows + i];
                }
                thread_offsets[tid + 1] += local_thread_row_contri_offsets[num_threads];
            }
        }

        for(int tid = 0; tid < num_threads; ++tid) {
            thread_offsets[tid + 1] += thread_offsets[tid];
        }
       
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            int offset = thread_offsets[tid];
            for(int i = start; i < end; ++i) {
                int* local_thread_row_contri_offsets = thread_row_contri_offsets.data() + i * (num_threads + 1);
                for(int t = 0; t < num_threads + 1; ++t) {
                    local_thread_row_contri_offsets[t] += offset;
                }
                offset = local_thread_row_contri_offsets[num_threads];
            }
        }
         
        row_contri.resize(all_contri, 0);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads) 
        for(int tid = 0; tid < num_threads; ++tid) {
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows); 
            int* local_row_contri_cnts_cnts = thread_row_contri_cnts_cnts.data() + tid * A->rows;
            for(int i = start; i < end; ++i) {
                int offset = A->row_ptr[i];
                int nnz = A->row_ptr[i + 1] - offset;
                int* col_idx = A->col_idx + offset;
                for(int j = 0; j < nnz; ++j) {
                    int idx = col_idx[j];
                    int &cnt = local_row_contri_cnts_cnts[idx];
                    int offset = thread_row_contri_offsets[idx * (num_threads + 1) + tid];
                    row_contri[offset + cnt] = i;
                    ++cnt;
                    assert(cnt <= thread_row_contri_offsets[idx * (num_threads + 1) + tid + 1] - thread_row_contri_offsets[idx * (num_threads + 1) + tid]);
                }  
            }
        } 
        vector<int> row_cnts(A->rows, 0);
        max_cnts.resize(num_threads, 0);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int i = start; i < end; ++i) {
                for(int t = 0; t < num_threads; ++t) {
                    row_cnts[i] += thread_row_cnts[t * A->rows + i];
                }
                if(row_cnts[i] > max_cnts[tid]) {
                    max_cnts[tid] = row_cnts[i];
                }
            }
        }
        thread_row_cnts.clear();
        thread_row_cnts.shrink_to_fit();
        thread_row_contri_cnts.clear();
        thread_row_contri_cnts.shrink_to_fit();
        thread_row_contri_cnts_cnts.clear();
        thread_row_contri_cnts_cnts.shrink_to_fit();

        row_contri_offsets.resize(A->rows + 1, 0);
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < A->rows; ++i) {
            row_contri_offsets[i] = thread_row_contri_offsets[i * (num_threads + 1)];
        }
        row_contri_offsets.back() = thread_row_contri_offsets.back();
        thread_row_contri_offsets.clear();
        thread_row_contri_offsets.shrink_to_fit();
         
        thread_offsets.clear();
        thread_offsets.resize(num_threads + 1, 0);
        B->rows = A->rows;
        B->rows_local = B->rows;
        B->rank = A->rank; 
        B->row_ptr = new int[A->rows + 1];
        B->row_ptr[0] = 0;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            vector<int> tmp_idx(max_cnts[tid]); 
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int i = start; i < end; ++i) {
                int contri_start = row_contri_offsets[i];
                int contri_end = row_contri_offsets[i + 1];
                int copy_offset = 0;
                for(int j = contri_start; j < contri_end; ++j) {
                    int row = row_contri[j];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - row_offset;
                    int* row_col_idx = A->col_idx + row_offset;  
                    memcpy(tmp_idx.data() + copy_offset, row_col_idx, row_nnz * sizeof(int)); 
                    copy_offset += row_nnz; 
                }
                std::sort(tmp_idx.begin(), tmp_idx.begin() + copy_offset, [] (int a, int b) ->bool { return a < b; });
                int out_row_nnz = std::unique(tmp_idx.begin(), tmp_idx.begin() + copy_offset) - tmp_idx.begin(); 
                B->row_ptr[i + 1] = out_row_nnz;
                thread_offsets[tid + 1] += out_row_nnz;
            }
        }

        for(int i = 0; i < num_threads; ++i) {
            thread_offsets[i + 1] += thread_offsets[i];
        }

        for(int i = 0; i < A->rows; ++i) {
            B->row_ptr[i + 1] += B->row_ptr[i];
        }
        B->nprocs = A->nprocs;
        B->nnz = thread_offsets.back();
        B->nnz_local = B->nnz;
        B->cols = A->cols;
        B->rows_offset = 0;
        B->col_idx = new int[B->nnz];
        B->value = new double[B->nnz]; 

#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            vector<int> tmp_idx(max_cnts[tid]); 
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int i = start; i < end; ++i) {
                int contri_start = row_contri_offsets[i];
                int contri_end = row_contri_offsets[i + 1];
                int copy_offset = 0;
                for(int j = contri_start; j < contri_end; ++j) {
                    int row = row_contri[j];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - row_offset; 
                    int* row_col_idx = A->col_idx + row_offset; 
                    memcpy(tmp_idx.data() + copy_offset, row_col_idx, row_nnz * sizeof(int)); 
                    copy_offset += row_nnz; 
                }
                std::sort(tmp_idx.begin(), tmp_idx.begin() + copy_offset, [] (int a, int b) ->bool { return a < b; });
                int* out_col_idx = B->col_idx + B->row_ptr[i];
                if(copy_offset > 0) {
                    out_col_idx[0] = tmp_idx[0];
                    int cnt = 1;
                    for(int j = 1; j < copy_offset; ++j) {
                        if(tmp_idx[j] != tmp_idx[j - 1]) {
                            out_col_idx[cnt++] = tmp_idx[j];
                        }
                    } 
                }
            }
        } 

        double t_start = omp_get_wtime();
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            vector<double> tmp_value(max_cnts[tid], 0.0); 
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int i = start; i < end; ++i) {
                int contri_start = row_contri_offsets[i];
                int contri_end = row_contri_offsets[i + 1]; 
                int out_row_offset = B->row_ptr[i];
                int out_nnz = B->row_ptr[i + 1] - out_row_offset;
                int *out_col_idx = B->col_idx + out_row_offset;
                double *out_value = B->value + out_row_offset;
                for(int j = contri_start; j < contri_end; ++j) {
                    int row = row_contri[j];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - row_offset; 
                    int* row_col_idx = A->col_idx + row_offset;
                    double* row_value = A->value + row_offset; 
                    int pos = std::lower_bound(row_col_idx, row_col_idx + row_nnz, i, [] (int a, int b) { return a < b; }) - row_col_idx;
                    assert(row_col_idx[pos] == i);
                    double value = row_value[pos];
#pragma omp simd
                    for(int k = 0; k < row_nnz; ++k) {
                        int idx = row_col_idx[k];
                        double value_2 = row_value[k];
                        int pos_2 = std::lower_bound(out_col_idx, out_col_idx + out_nnz, idx, [] (int a, int b) { return a < b; }) - out_col_idx;
                        out_value[pos_2] += value * value_2;
                    } 
                } 
            }
        }
        double t_end = omp_get_wtime(); 
    }
    else {
        delete [] B->value;
        B->value = new double[B->nnz];
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < B->nnz; ++i) {
            B->value[i] = 0.0;
        }
        int chunk_size = (A->rows + num_threads - 1) / num_threads;
        double t_start = omp_get_wtime();
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int tid = 0; tid < num_threads; ++tid) {
            int start = tid * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int i = start; i < end; ++i) {
                int contri_start = row_contri_offsets[i];
                int contri_end = row_contri_offsets[i + 1]; 
                int out_row_offset = B->row_ptr[i];
                int out_nnz = B->row_ptr[i + 1] - out_row_offset;
                int *out_col_idx = B->col_idx + out_row_offset;
                double *out_value = B->value + out_row_offset;
                for(int j = contri_start; j < contri_end; ++j) {
                    int row = row_contri[j];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - row_offset; 
                    int* row_col_idx = A->col_idx + row_offset;
                    double* row_value = A->value + row_offset; 
                    int pos = std::lower_bound(row_col_idx, row_col_idx + row_nnz, i, [] (int a, int b) { return a < b; }) - row_col_idx;
                    double value = row_value[pos];
#pragma omp simd
                    for(int k = 0; k < row_nnz; ++k) {
                        int idx = row_col_idx[k];
                        double value_2 = row_value[k];
                        int pos_2 = std::lower_bound(out_col_idx, out_col_idx + out_nnz, idx, [] (int a, int b) { return a < b; }) - out_col_idx;
                        out_value[pos_2] += value * value_2;
                    } 
                } 
            }
        }
        double t_end = omp_get_wtime(); 
    }
}
