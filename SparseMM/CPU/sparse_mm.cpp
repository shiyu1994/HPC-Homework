#include "sparse_mm.h"
#include <algorithm>
#include <set>
#include <map>
#include <omp.h>
#include <cassert>
#include <mpi.h>

using std::set;
using std::map;

SparseMM::SparseMM(CSR *AA, CSR *BB, int num_threadss) {
    A = AA;
    B = BB;
    num_threads = num_threadss;
}

void SparseMM::GatherData(bool is_first_time) { 
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
            vector<MPI_Request> rrecv(A_nprocs);
            MPI_Status status; 
            memcpy(A_value.data(),   A->value,   A->nnz_local*sizeof(double));
            for(int i = 1; i < A_nprocs; i ++) {  
                MPI_Irecv(A_value.data() + A_nnz_offsets[i],  A_nnz_offsets[i + 1] - A_nnz_offsets[i], MPI_DOUBLE, i, 4, MPI_COMM_WORLD, &rrecv[i]);       
            } 
            for(int i = 1; i < A_nprocs; ++i) {
                MPI_Wait(&rrecv[i], &status);
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
            MPI_Request rrecv, rsend; 
            MPI_Isend(A->value,   A->nnz_local, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD, &rsend);            
            MPI_Wait(&rsend, &status); 
            MPI_Irecv(A_value.data(), A_nnz, MPI_DOUBLE, 0, 7, MPI_COMM_WORLD, &rrecv);    
            MPI_Wait(&rrecv, &status);
        }
    }   
}


int SparseMM::GatherNNZ() {
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


void SparseMM::Compute0(bool is_first_time) {
    double t_start = omp_get_wtime();
    GatherData(is_first_time);
    double t_end = omp_get_wtime(); 
    MPI_Barrier(MPI_COMM_WORLD);
    if(is_first_time) {
        double t_start, t_end; 
        vector<vector<vector<int>>> thread_row_contri(num_threads);
        vector<vector<vector<int>>> thread_row_contri_pos(num_threads);
        int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            thread_row_contri[i].resize(A->rows_local);
            thread_row_contri_pos[i].resize(A->rows_local);
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int row_offset = A_row_ptr[j];
                int row_nnz = A_row_ptr[j + 1] - row_offset;
                int* row_col_idx = A_col_idx.data() + row_offset;
                for(int k = 0; k < row_nnz; ++k) {
                    int idx = row_col_idx[k];
                    if(idx >= A->rows_offset && idx < A->rows_offset + A->rows_local) {
                        thread_row_contri[i][idx - A->rows_offset].push_back(j);
                        thread_row_contri_pos[i][idx - A->rows_offset].push_back(k);
                    }
                }
            }
        }     
        t_start = omp_get_wtime();
        row_contri.resize(A->rows_local);
        row_contri_pos.resize(A->rows_local);
        vector<vector<int>> row_col_idx(A->rows_local);
        vector<int> thread_copy_offsets(num_threads + 1, 0);
        col_to_B_pos_map.resize(A->rows_local);
        chunk_size = (A->rows_local + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                set<int> nnz_cols;
                for(int tid = 0; tid < num_threads; ++tid) {
                    for(int k = 0; k < thread_row_contri[tid][j].size(); ++k) {
                        int row = thread_row_contri[tid][j][k];
                        int pos = thread_row_contri_pos[tid][j][k];
                        row_contri[j].push_back(row);
                        row_contri_pos[j].push_back(pos);
                        int row_offset = A_row_ptr[row];
                        int row_nnz = A_row_ptr[row + 1] - A_row_ptr[row];
                        int* col_idx = A_col_idx.data() + row_offset;
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
        for(int i = 0; i < num_threads; ++i) {
            thread_copy_offsets[i + 1] += thread_copy_offsets[i];
        }
        t_start = omp_get_wtime();
        B->rank = A->rank;
        B->nprocs = A->nprocs;
        B->rows = A->rows;
        B->cols = A->cols;
        B->rows_local = A->rows_local;
        B->rows_offset = A->rows_offset;
        B->nnz_local = thread_copy_offsets.back();
        B->nnz = B_nnz = GatherNNZ(); 
        B->row_ptr = new int[A->rows_local + 1];
        B->col_idx = new int[B->nnz_local];
        B->row_ptr[0] = row_nnz_offsets[A_rank];
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int cur_pos = thread_copy_offsets[i];
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                int row_nnz = row_col_idx[j].size();
                memcpy(B->col_idx + cur_pos, row_col_idx[j].data(), sizeof(int) * row_nnz);
                cur_pos += row_nnz;
                B->row_ptr[j + 1] = cur_pos + B->row_ptr[0]; 
            }
        }
        t_end = omp_get_wtime(); 
        B->value = new double[B->nnz_local];
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B->row_ptr[j] - B->row_ptr[0];
                double* B_row_value = B->value + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A_row_ptr[row];
                    int row_nnz = A_row_ptr[row + 1] - A_row_ptr[row];
                    int* row_col_idx = A_col_idx.data() + row_offset;
                    double* row_value = A_value.data() + row_offset;
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
    else {
        int chunk_size = (A->rows_local + num_threads - 1) / num_threads;  
        double t_start = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < B->nnz_local; ++i) {
            B->value[i] = 0.0;
        } 
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B->row_ptr[j] - B->row_ptr[0];
                double* B_row_value = B->value + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A_row_ptr[row];
                    int row_nnz = A_row_ptr[row + 1] - A_row_ptr[row];
                    int* row_col_idx = A_col_idx.data() + row_offset;
                    double* row_value = A_value.data() + row_offset;
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
        double t_end = omp_get_wtime(); 
    }
}

void SparseMM::Compute1(bool is_first_time) {
    double t_start = omp_get_wtime();
    GatherData(is_first_time);
    double t_end = omp_get_wtime(); 
    MPI_Barrier(MPI_COMM_WORLD);
    if(is_first_time) {
        double t_start, t_end; 
        vector<vector<vector<int>>> thread_row_contri(num_threads);
        vector<vector<vector<int>>> thread_row_contri_pos(num_threads);
        int chunk_size = (A->rows + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            thread_row_contri[i].resize(A->rows_local);
            thread_row_contri_pos[i].resize(A->rows_local);
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows);
            for(int j = start; j < end; ++j) {
                int row_offset = A_row_ptr[j];
                int row_nnz = A_row_ptr[j + 1] - row_offset;
                int* row_col_idx = A_col_idx.data() + row_offset;
                for(int k = 0; k < row_nnz; ++k) {
                    int idx = row_col_idx[k];
                    if(idx >= A->rows_offset && idx < A->rows_offset + A->rows_local) {
                        thread_row_contri[i][idx - A->rows_offset].push_back(j);
                        thread_row_contri_pos[i][idx - A->rows_offset].push_back(k);
                    }
                }
            }
        }     
        t_start = omp_get_wtime();
        row_contri.resize(A->rows_local);
        row_contri_pos.resize(A->rows_local);
        vector<vector<int>> row_col_idx(A->rows_local);
        vector<int> thread_copy_offsets(num_threads + 1, 0);
        col_to_B_pos_map.resize(A->rows_local);
        chunk_size = (A->rows_local + num_threads - 1) / num_threads;
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                set<int> nnz_cols;
                for(int tid = 0; tid < num_threads; ++tid) {
                    for(int k = 0; k < thread_row_contri[tid][j].size(); ++k) {
                        int row = thread_row_contri[tid][j][k];
                        int pos = thread_row_contri_pos[tid][j][k];
                        row_contri[j].push_back(row);
                        row_contri_pos[j].push_back(pos);
                        int row_offset = A_row_ptr[row];
                        int row_nnz = A_row_ptr[row + 1] - A_row_ptr[row];
                        int* col_idx = A_col_idx.data() + row_offset;
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
        for(int i = 0; i < num_threads; ++i) {
            thread_copy_offsets[i + 1] += thread_copy_offsets[i];
        }
        t_start = omp_get_wtime();
        B->rank = A->rank;
        B->nprocs = A->nprocs;
        B->rows = A->rows;
        B->cols = A->cols;
        B->rows_local = A->rows_local;
        B->rows_offset = A->rows_offset;
        B->nnz_local = thread_copy_offsets.back();
        B->nnz = B_nnz = GatherNNZ(); 
        B->row_ptr = new int[A->rows_local + 1];
        B->col_idx = new int[B->nnz_local];
        B->row_ptr[0] = row_nnz_offsets[A_rank];
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int cur_pos = thread_copy_offsets[i];
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                int row_nnz = row_col_idx[j].size();
                memcpy(B->col_idx + cur_pos, row_col_idx[j].data(), sizeof(int) * row_nnz);
                cur_pos += row_nnz;
                B->row_ptr[j + 1] = cur_pos + B->row_ptr[0]; 
            }
        }
        t_end = omp_get_wtime(); 
        B->value = new double[B->nnz_local];
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B->row_ptr[j] - B->row_ptr[0];
                double* B_row_value = B->value + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A_row_ptr[row];
                    int row_nnz = A_row_ptr[row + 1] - A_row_ptr[row];
                    int* row_col_idx = A_col_idx.data() + row_offset;
                    double* row_value = A_value.data() + row_offset;
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
    else {
        int chunk_size = (A->rows_local + num_threads - 1) / num_threads;  
        double t_start = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < B->nnz_local; ++i) {
            B->value[i] = 0.0;
        } 
#pragma omp parallel for schedule(static, 1) num_threads(num_threads)
        for(int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, A->rows_local);
            for(int j = start; j < end; ++j) {
                int B_row_offset = B->row_ptr[j] - B->row_ptr[0];
                double* B_row_value = B->value + B_row_offset;
                auto row_col_to_B_pos_map = col_to_B_pos_map[j];
                for(int k = 0; k < row_contri[j].size(); ++k) {
                    int row = row_contri[j][k];
                    int row_offset = A_row_ptr[row];
                    int row_nnz = A_row_ptr[row + 1] - A_row_ptr[row];
                    int* row_col_idx = A_col_idx.data() + row_offset;
                    double* row_value = A_value.data() + row_offset;
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
        double t_end = omp_get_wtime(); 
    }
}
