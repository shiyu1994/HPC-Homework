#include <iostream>
#include "stencil.hpp"
#include <vector>
#include <mpi.h>
#include <omp.h>

using std::vector;
using std::cout;
using std::endl;

CPUQueueStencil::CPUQueueStencil(const Config& configg, dvec64 &stencil, int block_x, int block_y, int block_t, int *argc, char ***argv):
Stencil(configg, stencil, block_x, block_y, block_t) {
    MPI_Init(argc, argv);
    z_size = (2 + y_dim) * (2 + x_dim);
    y_size = 2 + x_dim;
    all_z_dim = z_dim;
    all_steps = t_steps;
}

int CPUQueueStencil::InitMPI() { 
    MPI_Comm_size(MPI_COMM_WORLD, &num_machines);
    if(num_machines <= 1) {
        cout << "[CPU-Queue Stencil] please allocate more than 2 processes" << endl;
        exit(-1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank; 
}

void CPUQueueStencil::PartitionTasks() { 
    int layers_per_machine = (z_dim + num_machines + 1) / num_machines;
    z_comp_start = layers_per_machine * rank;
    z_comp_end = std::min(z_comp_start + layers_per_machine, z_dim + 2);
    z_stencil_start = std::max(0, z_comp_start - block_t - 1);
    z_stencil_end = std::min(z_comp_end + block_t + 1, z_dim + 2);
    if(rank == 0) {
        z_starts.resize(num_machines, 0);
        z_ends.resize(num_machines, 0);
        for(int i = 0; i < num_machines; ++i) {
            z_starts[i] = layers_per_machine * i;
            z_ends[i] = std::min(z_starts[i] + layers_per_machine, z_dim + 2);
        }
    }
}

void CPUQueueStencil::FinalizeMPI() {
    MPI_Finalize();
}

void CPUQueueStencil::InitData() {
    if(rank == 0) {
        Stencil::InitData(); 
    } 
}

void CPUQueueStencil::InitData(const dvec64 &data_init) {
    if(rank == 0) {
        Stencil::InitData(data_init);
    } 
}

void CPUQueueStencil::SendStencil() {
    if(rank == 0) {
        for(int i = 1; i < num_machines; ++i) {
            int send_start_z = std::max(0, z_starts[i] - block_t - 1);
            int send_end_z = std::min(z_ends[i] + block_t + 1, z_dim + 2);
            int size = z_size * (send_end_z - send_start_z); 
            MPI_Send(stencil.data() + send_start_z * z_size, size, MPI_DOUBLE, i, i, MPI_COMM_WORLD);
        }
    }
    else {
        int size = (z_stencil_end - z_stencil_start) * z_size;
        stencil.resize(size, 0.0); 
        stencil2.resize(size, 0.0);
        stencil3.resize(size, 0.0);
        MPI_Status status; 
        MPI_Recv(stencil.data(), size, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, &status);
    }
}

void CPUQueueStencil::SynchronizeStencil(double *s) {
    if(rank == 0) {
        MPI_Request request_send;
        int send_z_start = std::max(0, z_comp_end - block_t - 1);
        int send_z_end = z_comp_end;
        MPI_Isend(s + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request_send); 
        
        MPI_Request request;
        MPI_Status status;
        int recv_z_start = z_comp_end;
        int recv_z_end = z_stencil_end;
        MPI_Irecv(s + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request);
        
        MPI_Wait(&request, &status);
        MPI_Wait(&request_send, &status);
    }
    else if(rank == num_machines - 1) {
        MPI_Request request_send;
        int send_z_start = z_comp_start - z_stencil_start;
        int send_z_end = std::min(send_z_start + block_t + 1, z_dim + 2); 
        MPI_Isend(s + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &request_send);
        
        MPI_Request request;
        MPI_Status status;
        int recv_z_start = 0;
        int recv_z_end = block_t + 1;
        MPI_Irecv(s + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, &request);
        
        MPI_Wait(&request, &status);
        MPI_Wait(&request_send, &status);
    }
    else {
        MPI_Request request_send1, request_send2;
        int send_z_start = z_comp_start - z_stencil_start;
        int send_z_end = std::min(send_z_start + block_t + 1, z_dim + 2);
        MPI_Isend(s + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &request_send1);
        
        send_z_start = std::max(z_comp_end - z_stencil_start - block_t - 1, 0);
        send_z_end = std::min(send_z_start + block_t + 1, z_dim + 2);
        MPI_Isend(s + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, rank + 1, rank, MPI_COMM_WORLD, &request_send2);
        
        MPI_Request request1, request2;
        MPI_Status status;
        int recv_z_start = 0;
        int recv_z_end = block_t + 1;
        MPI_Irecv(s + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, &request1); 
        
        recv_z_start = z_comp_end - z_stencil_start;
        recv_z_end = std::min(recv_z_start + block_t + 1, z_dim + 2);
        MPI_Irecv(s + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, rank + 1, rank + 1, MPI_COMM_WORLD, &request2);
        
        MPI_Wait(&request1, &status); 
        MPI_Wait(&request2, &status);
        MPI_Wait(&request_send1, &status);
        MPI_Wait(&request_send2, &status);
    }
}

void CPUQueueStencil::GatherResult() {
    if(rank == 0) {
        MPI_Status status;
        for(int i = 1; i < num_machines; ++i) {
            int start = z_starts[i];
            int end = z_ends[i];
            MPI_Recv(stencil.data() + start * z_size, (end - start) * z_size, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
        }
    }
    else {
        MPI_Send(stencil.data() + (z_comp_start - z_stencil_start) * z_size, (z_comp_end - z_comp_start) * z_size, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    }
}

void CPUQueueStencil::Compute() { 
    double t_start = omp_get_wtime();
    PartitionTasks();
    MPI_Barrier(MPI_COMM_WORLD); 
    SendStencil(); 
    double t_end = omp_get_wtime();
    if(rank == 0) {
        cout << "distribute data time: " << (t_end - t_start) << endl;
    }

    //initialize cache
    cache.clear(); 
    cache.resize(num_threads);
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int tid = 0; tid < num_threads; ++tid) {  
        cache[tid].resize(block_t);
            for(int i = 0; i < block_t; ++i) {
            cache[tid][i].resize(3);
            for(int j = 0; j < 3; ++j) {
                cache[tid][i][j].resize(block_y);
                for(int k = 0; k < block_y; ++k) {
                    cache[tid][i][j][k].resize(block_x, 0.0);
                }
            }
        }    
    }


    z_dim = z_stencil_end - z_stencil_start - 2;
    t_steps = block_t + 1;
    int remain = all_steps % (block_t + 1);
    t_start = omp_get_wtime();
    double *s1 = stencil.data(), *s2 = stencil2.data(), *tmp_ptr = nullptr;
    
    for(int t = 0; t < all_steps - remain; t += (block_t + 1)) { 
        double t_start = omp_get_wtime();
        ComputeKernel(s1, s2);
        double t_end = omp_get_wtime(); 
        t_start = omp_get_wtime(); 
        SynchronizeStencil(s2);
        t_end = omp_get_wtime();
        //cout << "sync time: " << (t_end - t_start) << endl; 
        tmp_ptr = s1;
        s1 = s2;
        s2 = tmp_ptr;
    } 
    //copy back to stencil
    if(s1 != stencil.data()) {  
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < stencil.size(); ++i) {
            stencil[i] = s1[i];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD); 

    t_end = omp_get_wtime();
    if(rank == 0) {
        cout << "cpu queue kernel compute time: " << (t_end - t_start) << endl;
    }

    t_start = omp_get_wtime();
    GatherResult(); 
   
    t_end = omp_get_wtime();
    if(rank == 0) {
        cout << "gather result time: " << (t_end - t_start) << endl; 
    }
    z_dim = all_z_dim;
    t_steps = remain;
    if(rank == 0) {
        t_start = omp_get_wtime();
        Stencil::ComputeNaive();  
        t_end = omp_get_wtime();
        //cout << "compute remain time: " << (t_end - t_start) << endl;
    }
}

void CPUQueueStencil::NaiveComputeBound(double *s1, double *target, int z_size, int y_size, int steps) { 
    if(steps <= 1) {
        return;
    }

    double *s2 = stencil3.data();
    double *tmp = nullptr;
   
    if(rank == 0) {
    for(int t = 0; t < steps; ++t) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 1; i < z_dim + 1 - t; ++i) {
            for(int j = 1; j < 2 * steps - 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                }
            } 
            for(int j = 2 * steps - 1; j < y_dim + 1 - (2 * steps - 2); ++j) {
                for(int k = 1; k < 2 * steps - 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                        s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                            beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                            beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                }
                for(int k = x_dim + 1 - (2 * steps - 2); k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                        s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                            beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                            beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                } 
            }
            for(int j = y_dim + 1 - (2 * steps - 2); j < y_dim + 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                } 
            }
        }
        tmp = s1;
        s1 = s2;
        s2 = tmp;
    } 
    }
    else if(rank == num_machines - 1) {
    for(int t = 0; t < steps; ++t) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 1 + t; i < z_dim + 1; ++i) {
            for(int j = 1; j < 2 * steps - 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                }
            } 
            for(int j = 2 * steps - 1; j < y_dim + 1 - (2 * steps - 2); ++j) {
                for(int k = 1; k < 2 * steps - 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                        s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                            beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                            beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                }
                for(int k = x_dim + 1 - (2 * steps - 2); k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                        s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                            beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                            beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                } 
            }
            for(int j = y_dim + 1 - (2 * steps - 2); j < y_dim + 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                } 
            }
        }
        tmp = s1;
        s1 = s2;
        s2 = tmp;
    } 

    }
    else {
        for(int t = 0; t < steps; ++t) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 1 + t; i < z_dim + 1 - t; ++i) {
            for(int j = 1; j < 2 * steps - 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                }
            } 
            for(int j = 2 * steps - 1; j < y_dim + 1 - (2 * steps - 2); ++j) {
                for(int k = 1; k < 2 * steps - 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                        s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                            beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                            beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                }
                for(int k = x_dim + 1 - (2 * steps - 2); k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                        s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                            beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                            beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                } 
            }
            for(int j = y_dim + 1 - (2 * steps - 2); j < y_dim + 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                } 
            }
        }
        tmp = s1;
        s1 = s2;
        s2 = tmp;
    } 

    }

    double *source = s1;
    if(source != target) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 1; i < z_dim + 1; ++i) {
            for(int j = 1; j < steps; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    target[index] = source[index]; 
                }
            }
            for(int j = steps; j < y_dim - steps + 2; ++j) {
                for(int k = 1; k < steps; ++k) {
                    int index = i * z_size + j * y_size + k;
                    target[index] = source[index];
                }
                for(int k = x_dim - steps + 2; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    target[index] = source[index];
                }
            }
            for(int j = y_dim - steps + 2; j < y_dim + 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    target[index] = source[index];
                }
            }
        }
    }
}

void CPUQueueStencil::ComputeXYSubPlane(int x_start, int y_start, int z_size, int y_size, const double *s1, double *s2, vector<vector<vector<vector<double>>>> &cache) {
    int x_step = std::min(x_start + block_x, x_dim + 1) - x_start;
    int y_step = std::min(y_start + block_y, y_dim + 1) - y_start;  
 
    for(int zz = 0; zz < z_dim +  block_t; ++zz) {
        int z = zz; 
        if(z < z_dim) {
            vector<vector<double>> &plane0 = cache[0][z % 3];
            for(int y = 0; y < y_step; ++y) {
#pragma omp simd
                for(int x = 0; x < x_step; ++x) {
                    int index = (z + 1) * z_size + (y + y_start) * y_size + (x + x_start);
                    plane0[y][x] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] +
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] + 
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];                    
                }
            }            
        }
        for(int tt = 1; tt < block_t; ++tt) {
            z = zz - tt;
            if(z == 0) { 
                const vector<vector<double>> &plane_1 = cache[tt - 1][z % 3];
                const vector<vector<double>> &plane_2 = cache[tt - 1][(z + 1) % 3];
                vector<vector<double>> &plane = cache[tt][z % 3];
                for(int y = tt; y < y_step - tt; ++y) {
#pragma omp simd
                    for(int x = tt; x < x_step - tt; ++x) { 
                        plane[y][x] = alpha * plane_1[y][x] + beta_x_0 * plane_1[y][x - 1] + beta_x_1 * plane_1[y][x + 1] +
                            beta_y_0 * plane_1[y - 1][x] + beta_y_1 * plane_1[y + 1][x] +
                            beta_z_1 * plane_2[y][x];
                    }
                } 
            } 
            else if(z > 0 && z < z_dim - 1) {
                const vector<vector<double>> &plane_0 = cache[tt - 1][(z + 2) % 3];
                const vector<vector<double>> &plane_1 = cache[tt - 1][z % 3];
                const vector<vector<double>> &plane_2 = cache[tt - 1][(z + 1) % 3];
                vector<vector<double>> &plane = cache[tt][z % 3];
                for(int y = tt; y < y_step - tt; ++y) {
#pragma omp simd
                    for(int x = tt; x < x_step - tt; ++x) { 
                        plane[y][x] = alpha * plane_1[y][x] + beta_x_0 * plane_1[y][x - 1] + beta_x_1 * plane_1[y][x + 1] +
                            beta_y_0 * plane_1[y - 1][x] + beta_y_1 * plane_1[y + 1][x] +
                            beta_z_0 * plane_0[y][x] + beta_z_1 * plane_2[y][x];
                    }
                } 
            }
            else if(z == z_dim - 1) {
                const vector<vector<double>> &plane_1 = cache[tt - 1][z % 3];
                const vector<vector<double>> &plane_0 = cache[tt - 1][(z + 2) % 3];
                vector<vector<double>> &plane = cache[tt][z % 3];
                for(int y = tt; y < y_step - tt; ++y) {
#pragma omp simd
                    for(int x = tt; x < x_step - tt; ++x) { 
                        plane[y][x] = alpha * plane_1[y][x] + beta_x_0 * plane_1[y][x - 1] + beta_x_1 * plane_1[y][x + 1] +
                            beta_y_0 * plane_1[y - 1][x] + beta_y_1 * plane_1[y + 1][x] +
                            beta_z_0 * plane_0[y][x];
                    }
                } 
            }
        }
        z = zz - block_t;
        if(z == 0) { 
            const vector<vector<double>> &plane_1 = cache[block_t - 1][z % 3];
            const vector<vector<double>> &plane_2 = cache[block_t - 1][(z + 1) % 3];
            for(int y = block_t; y < y_step - block_t; ++y) {
#pragma omp simd
                for(int x = block_t; x < x_step - block_t; ++x) {
                    int index = (z + 1) * z_size + (y + y_start) * y_size + (x + x_start);
                    s2[index] = alpha * plane_1[y][x] + beta_x_0 * plane_1[y][x - 1] + beta_x_1 * plane_1[y][x + 1] +
                        beta_y_0 * plane_1[y - 1][x] + beta_y_1 * plane_1[y + 1][x] +
                        beta_z_1 * plane_2[y][x];
                }
            }
        }
        else if(z > 0 && z < z_dim - 1) {
            const vector<vector<double>> &plane_0 = cache[block_t - 1][(z + 2) % 3];
            const vector<vector<double>> &plane_1 = cache[block_t - 1][z % 3];
            const vector<vector<double>> &plane_2 = cache[block_t - 1][(z + 1) % 3];
            for(int y = block_t; y < y_step - block_t; ++y) {
#pragma omp simd
                for(int x = block_t; x < x_step - block_t; ++x) {
                    int index = (z + 1) * z_size + (y + y_start) * y_size + (x + x_start);
                    s2[index] = alpha * plane_1[y][x] + beta_x_0 * plane_1[y][x - 1] + beta_x_1 * plane_1[y][x + 1] +
                        beta_y_0 * plane_1[y - 1][x] + beta_y_1 * plane_1[y + 1][x] +
                        beta_z_0 * plane_0[y][x] + beta_z_1 * plane_2[y][x];
                }
            }
        }
        else if(z == z_dim - 1) {
            const vector<vector<double>> &plane_0 = cache[block_t - 1][(z + 2) % 3];
            const vector<vector<double>> &plane_1 = cache[block_t - 1][z % 3]; 
            for(int y = block_t; y < y_step - block_t; ++y) {
#pragma omp simd
                for(int x = block_t; x < x_step - block_t; ++x) {
                    int index = (z + 1) * z_size + (y + y_start) * y_size + (x + x_start);
                    s2[index] = alpha * plane_1[y][x] + beta_x_0 * plane_1[y][x - 1] + beta_x_1 * plane_1[y][x + 1] +
                        beta_y_0 * plane_1[y - 1][x] + beta_y_1 * plane_1[y + 1][x] +
                        beta_z_0 * plane_0[y][x];
                }
            }
        }
    }
}


void CPUQueueStencil::ComputeKernel(double *s1, double *s2) {
    //use 3.5d blocking
    int z_size = (y_dim + 2) * (x_dim + 2);
    int y_size = (x_dim + 2);

    int y_blocks = (y_dim + block_y - 2 * block_t - 1) / (block_y - 2 * block_t);
    int x_blocks = (x_dim + block_x - 2 * block_t - 1) / (block_x - 2 * block_t);
    double t_start = 0.0, t_end = 0.0;
    t_start = omp_get_wtime();
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for(int i = 0; i < y_blocks * x_blocks; ++i) {
        int j = i / x_blocks;
        int k = i % x_blocks;
        int x_start = 1 + k * (block_x - 2 * block_t);
        int y_start = 1 + j * (block_y - 2 * block_t);
        int tid = omp_get_thread_num();
        Stencil::ComputeXYSubPlane(x_start, y_start, z_size, y_size, s1, s2, cache[tid]); 
    }
    t_end = omp_get_wtime();
    //cout << "compute xy subplane time: " << (t_end - t_start) << endl;
    t_start = omp_get_wtime();
    NaiveComputeBound(s1, s2, z_size, y_size, block_t + 1); 
    t_end = omp_get_wtime(); 
    //cout << "naive compute bound time: " << (t_end - t_start) << endl; 
}
