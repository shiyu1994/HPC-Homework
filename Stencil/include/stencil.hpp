#include <iostream>
#include "config.h"
#include <vector>
#include "aligned_allocator.h"

using std::vector;

class Stencil {
protected:
    const Config &config;
    dvec64 &stencil;
    dvec64 stencil2;
    dvec64 stencil3;
    int x_dim, y_dim, z_dim;
    int num_threads;
    int t_steps;
    double alpha, beta_x_0, beta_x_1, beta_y_0, beta_y_1, beta_z_0, beta_z_1;
    int block_x, block_y, block_t;
   
    vector<vector<vector<vector<vector<double>>>>> cache; 
 
    void ComputeCPU();
    void ComputeXYSubPlane(int x_start, int y_start, int z_size, int y_size, const double *s1, double *s2, vector<vector<vector<vector<double>>>> &cache);
    void NaiveComputeBound(double *s1, double *target, int z_size, int y_size, int steps);
public:
    Stencil(const Config &configg, dvec64 &stencill, int block_x, int block_y, int block_t);

    virtual void InitData();
    virtual void InitData(const dvec64 &data_init);

    void ComputeNaive(); 

    virtual void Compute();
};

class GPUStencil : public Stencil {
protected:
    double *cuda_stencil1;
    double *cuda_stencil2;

    int block_dim_x, block_dim_y, grid_dim_x, grid_dim_y;
    
    void AllocGPUMem();
    void CopyDataFromCPUToGPU();
    void CopyResultFromGPUToCPU(); 
    void FreeGPUMem();
    //void InitData(const dvec64 &data_init);
    void CalcBlockAndGridDim();
public:
    virtual void Compute();
    GPUStencil(const Config &configg, dvec64 &stencil, int block_x, int block_y, int block_t);
};

class GPUCPUStencil : public GPUStencil {
private:    
    void CalcBlockAndGridDim(int block_t);
    void CPUCalcBoundary();
    void CopyBoundaryFromGPU(double *host_stencil, double *cuda_stencil);
    void CopyBoundaryToGPU(double *host_stencil, double *cuda_stencil, int y_size, int z_size, int steps);
    void CopyResultFromGPUToCPU(double *cuda_stencil, double* host_stencil, int block_t);
    void CopyGPUResult(double *host_stencil1, double *host_stencil2, int steps);
public:
    void Compute();
    GPUCPUStencil(const Config &configg, dvec64 &stencil, int block_x, int block_y, int block_t);
};

class CPUQueueStencil : public Stencil {
protected:
    int rank;
    int num_machines;
    int z_comp_start, z_comp_end;
    int z_stencil_start, z_stencil_end;
    int all_z_dim;
    int all_steps;
    vector<int> z_starts, z_ends;
    int z_size, y_size;
    void PartitionTasks(); 
    void SendStencil(); 
    void SynchronizeStencil(double *s);
    void GatherResult();
    void ComputeKernel(double *s1, double *s2);
    void NaiveComputeBound(double *s1, double *target, int z_size, int y_size, int steps);
    void ComputeXYSubPlane(int x_start, int y_start, int z_size, int y_size, const double *s1, double *s2, vector<vector<vector<vector<double>>>> &cache);

public:
    CPUQueueStencil(const Config &configg, dvec64 &stencil, int block_x, int block_y, int block_t, int *argc, char ***argv);
    int InitMPI();
    void FinalizeMPI();
    void InitData();
    void InitData(const dvec64 &data_init);
    void Compute(); 
};
