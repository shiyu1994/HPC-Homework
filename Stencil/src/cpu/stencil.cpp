#include "stencil.hpp"
#include "aligned_allocator.h"
#include "config.h"
#include <cstdlib>
#include <omp.h>
#include <iostream>
#include <cassert>

using std::cout;
using std::endl;

Stencil::Stencil(const Config& configg, dvec64 &stencill, int block_xx, int block_yy, int block_tt):
    config(configg), stencil(stencill) {
    srand(0);
    x_dim = config.x_dim;
    y_dim = config.y_dim;
    z_dim = config.z_dim;
    num_threads = config.num_threads;
    t_steps = config.t_steps;
    alpha = config.alpha;
    beta_x_0 = config.beta_x_0;
    beta_x_1 = config.beta_x_1;
    beta_y_0 = config.beta_y_0;
    beta_y_1 = config.beta_y_1;
    beta_z_0 = config.beta_z_0;
    beta_z_1 = config.beta_z_1;
    block_x = block_xx;
    block_y = block_yy;
    block_t = block_tt;
}

void Stencil::InitData() {
    stencil.resize((x_dim + 2) * (y_dim + 2) * (z_dim + 2), 0.0); 
    stencil2.resize((x_dim + 2) * (y_dim + 2) * (z_dim + 2), 0.0);   
    stencil3.resize((x_dim + 2) * (y_dim + 2) * (z_dim + 2), 0.0);
    int z_size = (y_dim + 2) * (x_dim + 2);
    int y_size = (x_dim + 2);
#pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 1; i < z_dim + 1; ++i) {
        for(int j = 1; j < y_dim + 1; ++j) {
            for(int k = 1; k < x_dim + 1; ++k) { 
                stencil[i * z_size + j * y_size + k] = std::rand() * 1.0 / RAND_MAX;
            } 
        }
    }
    //stencil[(x_dim +2) * (y_dim + 2) + y_dim+2 + 1] = 1.2;
}

void Stencil::InitData(const dvec64 &data_init) {
stencil.resize((x_dim + 2) * (y_dim + 2) * (z_dim + 2), 0.0); 
    stencil2.resize((x_dim + 2) * (y_dim + 2) * (z_dim + 2), 0.0);   
    stencil3.resize((x_dim + 2) * (y_dim + 2) * (z_dim + 2), 0.0);
    int z_size = (y_dim + 2) * (x_dim + 2);
    int y_size = (x_dim + 2);

#pragma omp parallel for schedule(static) num_threads(num_threads) 
    for(int i = 1; i < z_dim + 1; ++i) {
        for(int j = 1; j < y_dim + 1; ++j) {
            for(int k = 1; k < x_dim + 1; ++k) {
                stencil[i * z_size + j * y_size + k] = data_init[i * z_size + j * y_size + k]; 
            }
        }
    }
}

void Stencil::Compute() {    
     ComputeCPU();
}

void Stencil::ComputeNaive() { 
    int z_size = (y_dim + 2) * (x_dim + 2);
    int y_size = (x_dim + 2);

    double *s1 = stencil.data();
    double *s2 = stencil2.data();
    double *tmp = nullptr;
    for(int t = 0; t < t_steps; ++t) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 1; i < z_dim + 1; ++i) {
            for(int j = 1; j < y_dim + 1; ++j) {
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
    //copy back
    if(t_steps % 2 == 1) {
        for(int t = 0; t < t_steps; ++t) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
            for(int i = 1; i < z_dim + 1; ++i) {
                for(int j = 1; j < y_dim + 1; ++j) {
                    for(int k = 1; k < x_dim + 1; ++k) {
                        stencil[i * z_size + j * y_size + k] = stencil2[i * z_size + j * y_size + k];
                    }
                }
            }
        }
    }
}

void Stencil::ComputeXYSubPlane(int x_start, int y_start, int z_size, int y_size, const double *s1, double *s2, vector<vector<vector<vector<double>>>> &cache) {
    int x_step = std::min(x_start + block_x, x_dim + 1) - x_start;
    int y_step = std::min(y_start + block_y, y_dim + 1) - y_start;  

    for(int zz = 0; zz < z_dim +  block_t + 2; ++zz) {
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

void Stencil::ComputeCPU() {
    //use 3.5d blocking
    int z_size = (y_dim + 2) * (x_dim + 2);
    int y_size = (x_dim + 2);
    
    double *tmp_ptr = nullptr;
    double *s1 = stencil.data();
    double *s2 = stencil2.data();

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

    int remain = t_steps % (block_t + 1);
    for(int t = 0; t < t_steps - remain; t += block_t + 1) {        
        int y_blocks = (y_dim + block_y - 2 * block_t - 1) / (block_y - 2 * block_t);
        int x_blocks = (x_dim + block_x - 2 * block_t - 1) / (block_x - 2 * block_t);
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 0; i < y_blocks * x_blocks; ++i) {
            int j = i / x_blocks;
            int k = i % x_blocks;
            int x_start = 1 + k * (block_x - 2 * block_t);
            int y_start = 1 + j * (block_y - 2 * block_t);
            int tid = omp_get_thread_num();
            ComputeXYSubPlane(x_start, y_start, z_size, y_size, s1, s2, cache[tid]); 
        }
        NaiveComputeBound(s1, s2, z_size, y_size, block_t + 1);
        tmp_ptr = s1;
        s1 = s2;
        s2 = tmp_ptr;
    }
    for(int t = t_steps - remain; t < t_steps; ++t) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 1; i < z_dim + 1; ++i) {
            for(int j = 1; j < y_dim + 1; ++j) {
                for(int k = 1; k < x_dim + 1; ++k) {
                    int index = i * z_size + j * y_size + k;
                    s2[index] = alpha * s1[index] + beta_z_0 * s1[index - z_size] + beta_z_1 * s1[index + z_size] + 
                        beta_y_0 * s1[index - y_size] + beta_y_1 * s1[index + y_size] +
                        beta_x_0 * s1[index - 1] + beta_x_1 * s1[index + 1];
                }
            } 
        }
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
}

void Stencil::NaiveComputeBound(double *s1, double *target, int z_size, int y_size, int steps) { 
    if(steps <= 1) {
        return;
    }

    double *s2 = stencil3.data();
    double *tmp = nullptr;
    for(int t = 0; t < steps; ++t) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
        for(int i = 1; i < z_dim + 1; ++i) {
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
