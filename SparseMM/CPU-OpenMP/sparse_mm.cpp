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
    if(is_first_time) {
        //double t_start, t_end;
        //t_start = omp_get_wtime();
        //cout << "step 0" << endl;
        vector<vector<vector<int>>> thread_row_contri(num_threads); 
        int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            thread_row_contri[i].resize(A->rows);
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int row_offset = A->row_ptr[j];
                int row_nnz = A->row_ptr[j + 1] - row_offset;
                int* row_col_idx = A->col_idx + row_offset;
                for(int k = 0; k < row_nnz; ++k) {
                    thread_row_contri[i][row_col_idx[k]].push_back(j); 
                }
            }
        }    
        //t_end = omp_get_wtime();
        //cout << "step 1 " << (t_end - t_start) << endl;
        //t_start = omp_get_wtime();
        row_contri.resize(A->rows); 
        vector<vector<int>> row_col_idx(A->rows);
        vector<int> thread_copy_offsets(num_threads + 1, 0);
        col_to_B_pos_map.resize(A->rows);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                set<int> nnz_cols;
                for(int tid = 0; tid < num_threads; ++tid) {
                    for(int k = 0; k < thread_row_contri[tid][j].size(); ++k) {
                        int row = thread_row_contri[tid][j][k];
                        row_contri[j].push_back(row); 
                        int row_offset = A->row_ptr[row];
                        int row_nnz = A->row_ptr[row + 1] - A->row_ptr[row];
                        int* col_idx = A->col_idx + row_offset;
                        for(int z = 0; z < row_nnz; ++z) {
                            nnz_cols.insert(col_idx[z]);
                        }
                    }
                }
                auto p = nnz_cols.begin();
                int k = 0;
                int out_row_nnz = nnz_cols.size();
                thread_copy_offsets[i + 1] += out_row_nnz;
                row_col_idx[j].resize(out_row_nnz);
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(p = nnz_cols.begin(), k = 0; p != nnz_cols.end(); ++k, ++p) {
                    row_col_idx[j][k] = *p;
                    row_col_to_B_pos_map[*p] = k;
                }
            }
        }
        //t_end = omp_get_wtime();
        //cout << "step 2 " << (t_end - t_start) << endl;
        for(int i = 0; i < num_threads; ++i) {
            thread_copy_offsets[i + 1] += thread_copy_offsets[i];
        }
        //t_start = omp_get_wtime();
        B_nnz = thread_copy_offsets.back();
        B_row_ptr.resize(A->rows + 1, 0);
        B_col_idx.resize(B_nnz);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int cur_pos = thread_copy_offsets[i];
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int row_nnz = row_col_idx[j].size();
                memcpy(B_col_idx.data() + cur_pos, row_col_idx[j].data(), sizeof(int) * row_nnz);
                cur_pos += row_nnz;
                B_row_ptr[j + 1] = cur_pos; 
            }
        }
        //t_end = omp_get_wtime();
        //cout << "step 3 " << (t_end - t_start) << endl;
        B_value.resize(B_nnz, 0.0);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B_row_ptr[j];
                double* B_row_value = B_value.data() + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - A->row_ptr[row];
                    int* row_col_idx = A->col_idx + row_offset;
                    double* row_value = A->value + row_offset;
                    int pos = std::lower_bound(row_col_idx, row_col_idx + row_nnz, j, [] (int a, int b) -> bool {return a < b;}) - row_col_idx;
                    assert(pos >= 0 && pos < row_nnz && row_col_idx[pos] == j);
                    double value = row_value[pos];
                    for(int z = 0; z < row_nnz; ++z) {
                        int idx = row_col_idx[z];
                        double value2 = row_value[z];
                        int B_col_pos = row_col_to_B_pos_map[idx];
                        B_row_value[B_col_pos] += value * value2;
                    }
                }
            }
        } 
        B->rank = A->rank;
        B->nprocs = 1;
        B->rows = A->rows;
        B->cols = A->cols;
        B->nnz = B_nnz;
        B->rows_local = A->rows;
        B->rows_offset = 0;
        B->nnz_local = B_nnz;
        B->row_ptr = new int[A->rows + 1];
        B->col_idx = new int[B_nnz];
        B->value = new double[B_nnz]; 
        memcpy(B->row_ptr, B_row_ptr.data(), sizeof(int) * (A->rows + 1));
        memcpy(B->col_idx, B_col_idx.data(), sizeof(int) * B_nnz);
        memcpy(B->value, B_value.data(), sizeof(double) * B_nnz); 
    }

    else {
        int chunk_size = (A->rows + num_threads - 1) / num_threads;  
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B_row_ptr[j];
                double* B_row_value = B->value + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - A->row_ptr[row];
                    int* row_col_idx = A->col_idx + row_offset;
                    double* row_value = A->value + row_offset;
                    int pos = std::lower_bound(row_col_idx, row_col_idx + row_nnz, j, [] (int a, int b) -> bool { return a < b; }) - row_col_idx; 
                    double value = row_value[pos];
                    for(int z = 0; z < row_nnz; ++z) {
                        int idx = row_col_idx[z];
                        double value2 = row_value[z];
                        int B_col_pos = row_col_to_B_pos_map[idx];
                        B_row_value[B_col_pos] += value * value2;
                    }
                }
            }
        } 
    }
}

void SparseMM::Compute1(bool is_first_time) {
    if(is_first_time) {
        double t_start, t_end;
        //t_start = omp_get_wtime();
        //cout << "step 0" << endl;
        vector<vector<vector<int>>> thread_row_contri(num_threads);
        vector<vector<vector<int>>> thread_row_contri_pos(num_threads);
        int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            thread_row_contri[i].resize(A->rows);
            thread_row_contri_pos[i].resize(A->rows);
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int row_offset = A->row_ptr[j];
                int row_nnz = A->row_ptr[j + 1] - row_offset;
                int* row_col_idx = A->col_idx + row_offset;
                for(int k = 0; k < row_nnz; ++k) {
                    thread_row_contri[i][row_col_idx[k]].push_back(j);
                    thread_row_contri_pos[i][row_col_idx[k]].push_back(k);
                }
            }
        }    
        //t_end = omp_get_wtime();
        //cout << "step 1 " << (t_end - t_start) << endl;
        t_start = omp_get_wtime();
        row_contri.resize(A->rows);
        row_contri_pos.resize(A->rows);
        vector<vector<int>> row_col_idx(A->rows);
        vector<int> thread_copy_offsets(num_threads + 1, 0);
        col_to_B_pos_map.resize(A->rows);
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                set<int> nnz_cols;
                for(int tid = 0; tid < num_threads; ++tid) {
                    for(int k = 0; k < thread_row_contri[tid][j].size(); ++k) {
                        int row = thread_row_contri[tid][j][k];
                        int pos = thread_row_contri_pos[tid][j][k];
                        row_contri[j].push_back(row);
                        row_contri_pos[j].push_back(pos);
                        int row_offset = A->row_ptr[row];
                        int row_nnz = A->row_ptr[row + 1] - A->row_ptr[row];
                        int* col_idx = A->col_idx + row_offset;
                        for(int z = 0; z < row_nnz; ++z) {
                            nnz_cols.insert(col_idx[z]);
                        }
                    }
                }
                auto p = nnz_cols.begin();
                int k = 0;
                int out_row_nnz = nnz_cols.size();
                thread_copy_offsets[i + 1] += out_row_nnz;
                row_col_idx[j].resize(out_row_nnz);
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(p = nnz_cols.begin(), k = 0; p != nnz_cols.end(); ++k, ++p) {
                    row_col_idx[j][k] = *p;
                    row_col_to_B_pos_map[*p] = k;
                }
            }
        }
        t_end = omp_get_wtime();
        cout << "step 2 " << (t_end - t_start) << endl;
        for(int i = 0; i < num_threads; ++i) {
            thread_copy_offsets[i + 1] += thread_copy_offsets[i];
        }
        t_start = omp_get_wtime();
        B->nnz = B_nnz = thread_copy_offsets.back();
        B->row_ptr = new int[A->rows + 1];
        B->col_idx = new int[B_nnz];
        B->row_ptr[0] = 0;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int cur_pos = thread_copy_offsets[i];
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int row_nnz = row_col_idx[j].size();
                memcpy(B->col_idx + cur_pos, row_col_idx[j].data(), sizeof(int) * row_nnz);
                cur_pos += row_nnz;
                B->row_ptr[j + 1] = cur_pos; 
            }
        }
        t_end = omp_get_wtime();
        cout << "step 3 " << (t_end - t_start) << endl;
        B->value = new double[B_nnz];
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B->row_ptr[j];
                double* B_row_value = B->value + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - A->row_ptr[row];
                    int* row_col_idx = A->col_idx + row_offset;
                    double* row_value = A->value + row_offset;
                    int pos = row_contri_pos[j][k]; 
                    double value = row_value[pos];
#pragma omp simd
                    for(int z = 0; z < row_nnz; ++z) {
                        int idx = row_col_idx[z];
                        double value2 = row_value[z];
                        int B_col_pos = row_col_to_B_pos_map[idx];
                        B_row_value[B_col_pos] += value * value2;
                    }
                }
            }
        } 
        B->rank = A->rank;
        B->nprocs = 1;
        B->rows = A->rows;
        B->cols = A->cols;
        B->nnz = B_nnz;
        B->rows_local = A->rows;
        B->rows_offset = 0;
        B->nnz_local = B->nnz;
        //B->row_ptr = new int[A->rows + 1];
        //B->col_idx = new int[B->nnz];
        //B->value = new double[B->nnz]; 
        //memcpy(B->row_ptr, B_row_ptr.data(), sizeof(int) * (A->rows + 1));
        //memcpy(B->col_idx, B_col_idx.data(), sizeof(int) * B_nnz);
        //memcpy(B->value, B_value.data(), sizeof(double) * B_nnz); 
    }
    else {
        int chunk_size = (A->rows + num_threads - 1) / num_threads;  
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B->row_ptr[j];
                double* B_row_value = B->value + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A->row_ptr[row];
                    int row_nnz = A->row_ptr[row + 1] - A->row_ptr[row];
                    int* row_col_idx = A->col_idx + row_offset;
                    double* row_value = A->value + row_offset;
                    int pos = row_contri_pos[j][k];
                    double value = row_value[pos];
#pragma omp simd
                    for(int z = 0; z < row_nnz; ++z) {
                        int idx = row_col_idx[z];
                        double value2 = row_value[z];
                        int B_col_pos = row_col_to_B_pos_map[idx];
                        B_row_value[B_col_pos] += value * value2;
                    }
                }
            }
        } 
    }
}
