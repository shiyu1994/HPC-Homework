#include <iostream>
#include "stencil.hpp"
#include <vector>
#include <mpi.h>

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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank; 
}

void CPUQueueStencil::PartitionTasks() {
    int layers_per_machine = (z_dim + num_machines + 1) / num_machines;
    z_comp_start = layers_per_machine * rank;
    z_comp_end = std::min(z_comp_start + layers_per_machine, z_dim + 2);
    z_stencil_start = std::max(0, z_comp_start - block_t);
    z_stencil_end = std::min(z_comp_start + block_t, z_dim + 2);
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
            int send_start_z = std::max(0, z_starts[i] - block_t);
            int send_end_z = std::min(z_ends[i] + block_t, z_dim + 2);
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

void CPUQueueStencil::SynchronizeStencil() {
    if(rank == 0) {
        MPI_Request request_send;
        int send_z_start = std::max(0, z_comp_end - block_t);
        int send_z_end = z_comp_end;
        MPI_Isend(stencil.data() + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request_send); 
        
        MPI_Request request;
        MPI_Status status;
        int recv_z_start = z_comp_end;
        int recv_z_end = z_stencil_end;
        MPI_Irecv(stencil.data() + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request);
        
        MPI_Wait(&request_send, &status);
        MPI_Wait(&request, &status);
    }
    else if(rank == num_machines - 1) {
        MPI_Request request_send;
        int send_z_start = z_comp_start - z_stencil_start;
        int send_z_end = std::min(send_z_start + block_t, z_dim + 2); 
        MPI_Isend(stencil.data() + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &request_send);
        
        MPI_Request request;
        MPI_Status status;
        int recv_z_start = 0;
        int recv_z_end = block_t;
        MPI_Irecv(stencil.data() + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, &request);
        
        MPI_Wait(&request_send, &status);
        MPI_Wait(&request, &status);
    }
    else {
        MPI_Request request_send1, request_send2;
        int send_z_start = z_comp_start - z_stencil_start;
        int send_z_end = std::min(send_z_start + block_t, z_dim + 2);
        MPI_Isend(stencil.data() + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &request_send1);
        
        send_z_start = std::max(z_comp_end - z_stencil_start - block_t, 0);
        send_z_end = std::min(send_z_start + block_t, z_dim + 2);
        MPI_Isend(stencil.data() + send_z_start * z_size, (send_z_end - send_z_start) * z_size, MPI_DOUBLE, rank + 1, rank, MPI_COMM_WORLD, &request_send2);
        
        MPI_Request request1, request2;
        MPI_Status status;
        int recv_z_start = 0;
        int recv_z_end = block_t;
        MPI_Irecv(stencil.data() + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, &request1); 
        
        recv_z_start = z_comp_end - z_stencil_start;
        recv_z_end = std::min(recv_z_start + block_t, z_dim + 2);
        MPI_Irecv(stencil.data() + recv_z_start * z_size, (recv_z_end - recv_z_start) * z_size, MPI_DOUBLE, rank + 1, rank + 1, MPI_COMM_WORLD, &request2);
        
        MPI_Wait(&request_send1, &status);
        MPI_Wait(&request_send2, &status);
        MPI_Wait(&request1, &status); 
        MPI_Wait(&request2, &status);
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
    z_dim = z_stencil_end - z_stencil_start - 2;
    t_steps = block_t;
    int remain = all_steps % block_t;
    for(int t = 0; t < all_steps - remain; t += block_t) {
        Stencil::Compute();
        SynchronizeStencil();
    }
    GatherResult(); 
    z_dim = all_z_dim;
    t_steps = remain;
    Stencil::ComputeNaive(); 
}
