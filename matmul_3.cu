#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->


    // OPTIONAL: Uncomment this block to include your kernel implementation
    // from Lab 5 for easy comparison.

    ////////////////////////////////////////////////////////////////////////////////
    // Optimized GPU Implementation with Reduction along k (Baseline from Lab 5)

    #define HAS_LAB_5_BASELINE_IMPL // <~~ keep this line if you want to benchmark your Lab 5 kernel!


namespace matmul_improved {

__global__ void __launch_bounds__(256) matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    
    constexpr int32_t TILE_DIM = 64;
    constexpr int32_t THREAD_BLOCK_DIM = 16;
    constexpr int32_t REG_TILE_DIM = TILE_DIM / THREAD_BLOCK_DIM; 
    
    float reg_accumulator[REG_TILE_DIM][REG_TILE_DIM];
    
    
    
    #pragma unroll 4
    for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {
        #pragma unroll 4
        for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {
            reg_accumulator[row_idx][col_idx] = 0;
        }
    }

    constexpr int32_t PADDING = 8;
    __shared__ float smem_a[TILE_DIM][TILE_DIM + PADDING];
    __shared__ float smem_b[TILE_DIM][TILE_DIM + PADDING];
    const int32_t out_row_start = blockIdx.y * TILE_DIM;
    const int32_t out_col_start = blockIdx.x * TILE_DIM;
    const int32_t thread_x = threadIdx.x;
    const int32_t thread_y = threadIdx.y;
    const int32_t flat_thread_id = thread_y * THREAD_BLOCK_DIM + thread_x;
    
    
    for (int32_t k_tile_idx = 0; k_tile_idx < (size_k + TILE_DIM - 1) / TILE_DIM; ++k_tile_idx) {
        const int32_t k_offset = k_tile_idx * TILE_DIM;
        
        
        // Load A into smem
        if ((out_row_start + TILE_DIM <= size_i) && (k_offset + TILE_DIM <= size_k)) {
            #pragma unroll
            for (int32_t iter = 0; iter < REG_TILE_DIM; ++iter) {
                
                const int32_t smem_row = (flat_thread_id + iter * 256) / (TILE_DIM / 4);
                const int32_t smem_col_base =  ((flat_thread_id + iter * 256) % (TILE_DIM / 4)) * 4;
                const int32_t row = out_row_start + smem_row;
                const int32_t col = k_offset + smem_col_base;
                
                const float4 loaded_vec = *reinterpret_cast<const float4*>(&a[row * size_k + col]);
                smem_a[smem_row][smem_col_base + 0] = loaded_vec.x;
                smem_a[smem_row][smem_col_base + 1] = loaded_vec.y;
                smem_a[smem_row][smem_col_base + 2] = loaded_vec.z;
                smem_a[smem_row][smem_col_base + 3] = loaded_vec.w;
            }
        } 

        else {
            #pragma unroll
            for (int32_t iter = 0; iter < REG_TILE_DIM; ++iter) {
                const int32_t smem_row = (flat_thread_id + iter * 256) / (TILE_DIM / 4);
                const int32_t smem_col_base = ((flat_thread_id + iter * 256) % (TILE_DIM / 4)) * 4;
                
                const int32_t row = out_row_start + smem_row;
                const int32_t col = k_offset + smem_col_base;
                
                if (row < size_i && col + 3 < size_k) {
                    const float4 loaded_vec = *reinterpret_cast<const float4*>(&a[row * size_k + col]);
                    smem_a[smem_row][smem_col_base + 0] = loaded_vec.x;
                    smem_a[smem_row][smem_col_base + 1] = loaded_vec.y;
                    smem_a[smem_row][smem_col_base + 2] = loaded_vec.z;
                    smem_a[smem_row][smem_col_base + 3] = loaded_vec.w;
                }
                else {
                    for (int elem_idx = 0; elem_idx < 4; ++elem_idx) {
                        const int32_t smem_col = smem_col_base + elem_idx;
                        const int32_t global_col = col + elem_idx;
                        smem_a[smem_row][smem_col] = (row < size_i && global_col < size_k) ? a[row * size_k + global_col] : 0.0f;
                    }
                }
            }
        }

        // Load B into smem
        if ((out_col_start + TILE_DIM <= size_j) && (k_offset + TILE_DIM <= size_k)) {
            #pragma unroll
            for (int32_t iter = 0; iter < 4; ++iter) {
                const int32_t smem_row = (flat_thread_id + iter * 256) / (TILE_DIM / 4);
                const int32_t smem_col_base = ((flat_thread_id + iter * 256) % (TILE_DIM / 4)) * 4;
                
                const int32_t global_row = k_offset + smem_row;
                const int32_t global_col = out_col_start + smem_col_base;
                
                const float4 loaded_vec = *reinterpret_cast<const float4*>(&b[global_row * size_j + global_col]);
                smem_b[smem_row][smem_col_base + 0] = loaded_vec.x;
                smem_b[smem_row][smem_col_base + 1] = loaded_vec.y;
                smem_b[smem_row][smem_col_base + 2] = loaded_vec.z;
                smem_b[smem_row][smem_col_base + 3] = loaded_vec.w;
            }
        } 
        
        else {
            #pragma unroll
            for (int32_t iter = 0; iter < 4; ++iter) {
                const int32_t smem_row = (flat_thread_id + iter * 256) / (TILE_DIM / 4);
                const int32_t smem_col_base = ((flat_thread_id + iter * 256) % (TILE_DIM / 4)) * 4;
                
                const int32_t row = k_offset + smem_row;
                const int32_t col = out_col_start + smem_col_base;
                
                if (row < size_k && col + 3 < size_j) {
                    const float4 loaded_vec = *reinterpret_cast<const float4*>(&b[row * size_j + col]);
                    smem_b[smem_row][smem_col_base + 0] = loaded_vec.x;
                    smem_b[smem_row][smem_col_base + 1] = loaded_vec.y;
                    smem_b[smem_row][smem_col_base + 2] = loaded_vec.z;
                    smem_b[smem_row][smem_col_base + 3] = loaded_vec.w;
                } 
                else {
                    for (int elem_idx = 0; elem_idx < 4; ++elem_idx) {
                        const int32_t smem_col = smem_col_base + elem_idx;
                        const int32_t global_col = col + elem_idx;
                        smem_b[smem_row][smem_col] = (row < size_k && global_col < size_j) ? b[row * size_j + global_col] : 0.0f;
                    }
                }
            }
        }
        
        __syncthreads();
        

        //outer product
        #pragma unroll 8
        for (int32_t dot_idx = 0; dot_idx < TILE_DIM; ++dot_idx) {
            float a_fragment[REG_TILE_DIM];
            float b_fragment[REG_TILE_DIM];

            #pragma unroll
            for (int32_t idx = 0; idx < REG_TILE_DIM; ++idx) {
                a_fragment[idx] = smem_a[thread_y * REG_TILE_DIM + idx][dot_idx];
                b_fragment[idx] = smem_b[dot_idx][thread_x * REG_TILE_DIM + idx];
            }
            
                
            #pragma unroll
            for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {

                #pragma unroll

                for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {
                    reg_accumulator[row_idx][col_idx] += a_fragment[row_idx] * b_fragment[col_idx];
                }
            }
        }
        
        __syncthreads();
    }
    
    //write the output
    if ((out_row_start + TILE_DIM <= size_i) && (out_col_start + TILE_DIM <= size_j)) {
        #pragma unroll

        for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {
            #pragma unroll
            for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {
                const int32_t global_row = out_row_start + thread_y * REG_TILE_DIM + row_idx;
                const int32_t global_col = out_col_start + thread_x * REG_TILE_DIM + col_idx;
                c[global_row * size_j + global_col] = reg_accumulator[row_idx][col_idx];
            }
        }
    } 
    else {

        #pragma unroll
        for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {
            #pragma unroll
            for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {

                const int32_t global_col = out_col_start + thread_x * REG_TILE_DIM + col_idx;
                const int32_t global_row = out_row_start + thread_y * REG_TILE_DIM + row_idx;
                

                if (global_row < size_i && global_col < size_j) {
                    c[global_row * size_j + global_col] = reg_accumulator[row_idx][col_idx];
                }
            }
        }
    }
}

void launch_matmul_improved(
int32_t size_i,
int32_t size_j,
int32_t size_k,
float const *a, /* pointer to GPU memory */
float const *b, /* pointer to GPU memory */
float *c /* pointer to GPU memory */) {

constexpr int32_t TILE_SIZE = 64;
constexpr int32_t BLOCK_SIZE = 16;

dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

dim3 grid((size_j + TILE_SIZE - 1) / TILE_SIZE, (size_i + TILE_SIZE - 1) / TILE_SIZE);

matmul_improved<<<grid, threads>>>(size_i, size_j, size_k, a, b, c);
}
}; // namespace matmul_improved
namespace matmul_improved_reduce {



/* TODO: your GPU kernels here... */

__global__ void __launch_bounds__(256) matmul_splitk(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *partial_results,
    int32_t elements_per_partition) {
    
    constexpr int32_t TILE_DIM = 64;
    constexpr int32_t THREADS_PER_DIM = 16;
    constexpr int32_t REG_TILE_DIM = TILE_DIM / THREADS_PER_DIM;
    float reg_accumulator[REG_TILE_DIM][REG_TILE_DIM];

    
    
    #pragma unroll
    for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {
        #pragma unroll
        for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {
            reg_accumulator[row_idx][col_idx] = 0.0f;
        }
    }

    const int32_t partition_id = blockIdx.z;
    const int32_t k_begin = partition_id * elements_per_partition;
    const int32_t k_limit = min(k_begin + elements_per_partition, size_k);
    constexpr int32_t PADDING = 8;
    __shared__ float smem_a[TILE_DIM][TILE_DIM + PADDING];
    __shared__ float smem_b[TILE_DIM][TILE_DIM + PADDING];
    const int32_t thread_x = threadIdx.x;
    const int32_t thread_y = threadIdx.y;
    const int32_t flat_thread_id = thread_y * THREADS_PER_DIM + thread_x;
    const int32_t out_row_start = blockIdx.y * TILE_DIM;
    const int32_t out_col_start = blockIdx.x * TILE_DIM;
    


    
    const int32_t first_tile = k_begin / TILE_DIM;
    const int32_t last_tile = (k_limit - 1) / TILE_DIM;
    
    for (int32_t tile_idx = first_tile; tile_idx <= last_tile; ++tile_idx) {
        const int32_t tile_k_begin = tile_idx * TILE_DIM;
        const int32_t tile_k_limit = (tile_idx + 1) * TILE_DIM;
        
        // Load A into smem
        if ((out_row_start + TILE_DIM <= size_i) &&(tile_k_limit <= size_k)) {
            #pragma unroll
            for (int32_t iter = 0; iter < REG_TILE_DIM; ++iter) {
                const int32_t linear_idx = flat_thread_id + iter * 256;
                const int32_t smem_row = linear_idx / (TILE_DIM / 4);
                const int32_t smem_col_base = (linear_idx % (TILE_DIM / 4)) * 4;
                const int32_t row = out_row_start + smem_row;
                const int32_t col = tile_idx * TILE_DIM + smem_col_base;
                
                const float4 loaded_vec = *reinterpret_cast<const float4*>(&a[row * size_k + col]);
                smem_a[smem_row][smem_col_base + 0] = loaded_vec.x;
                smem_a[smem_row][smem_col_base + 1] = loaded_vec.y;
                smem_a[smem_row][smem_col_base + 2] = loaded_vec.z;
                smem_a[smem_row][smem_col_base + 3] = loaded_vec.w;
            }
        } 

        else {
            #pragma unroll
            for (int32_t iter = 0; iter < 4; ++iter) {
                const int32_t linear_idx = flat_thread_id + iter * 256;
                const int32_t smem_row = linear_idx / (TILE_DIM / 4);
                const int32_t smem_col_base = (linear_idx % (TILE_DIM / 4)) * 4;
                
                const int32_t row = out_row_start + smem_row;
                const int32_t col = tile_idx * TILE_DIM + smem_col_base;
                
                if (row < size_i && col + 3 < size_k) {
                    const float4 loaded_vec = *reinterpret_cast<const float4*>(&a[row * size_k + col]);
                    smem_a[smem_row][smem_col_base + 0] = loaded_vec.x;
                    smem_a[smem_row][smem_col_base + 1] = loaded_vec.y;
                    smem_a[smem_row][smem_col_base + 2] = loaded_vec.z;
                    smem_a[smem_row][smem_col_base + 3] = loaded_vec.w;
                } 
                else {
                    for (int elem_idx = 0; elem_idx < 4; ++elem_idx) {
                        const int32_t smem_col = smem_col_base + elem_idx;
                        const int32_t global_col = col + elem_idx;
                        smem_a[smem_row][smem_col] = (row < size_i && col + elem_idx < size_k) ? a[row * size_k + global_col] : 0.0f;
                    }
                }
            }
        }
        
        // Load B into smem
        if ((out_col_start + TILE_DIM <= size_j) && (tile_k_limit <= size_k)) {
            #pragma unroll
            for (int32_t iter = 0; iter < 4; ++iter) {
                const int32_t linear_idx = flat_thread_id + iter * 256;
                const int32_t smem_row = linear_idx / (TILE_DIM / 4);
                const int32_t smem_col_base = (linear_idx % (TILE_DIM / 4)) * 4;
                
                const int32_t global_row = tile_idx * TILE_DIM + smem_row;
                const int32_t global_col = out_col_start + smem_col_base;
                
                const float4 loaded_vec = *reinterpret_cast<const float4*>(&b[global_row * size_j + global_col]);
                smem_b[smem_row][smem_col_base + 0] = loaded_vec.x;
                smem_b[smem_row][smem_col_base + 1] = loaded_vec.y;
                smem_b[smem_row][smem_col_base + 2] = loaded_vec.z;
                smem_b[smem_row][smem_col_base + 3] = loaded_vec.w;
            }
        }
        else {
            #pragma unroll
            for (int32_t iter = 0; iter < 4; ++iter) {
                const int32_t linear_idx = flat_thread_id + iter * 256;
                const int32_t smem_row = linear_idx / (TILE_DIM / 4);
                const int32_t smem_col_base = (linear_idx % (TILE_DIM / 4)) * 4;
                
                const int32_t row = tile_idx * TILE_DIM + smem_row;
                const int32_t col = out_col_start + smem_col_base;
                
                if (row < size_k && col + 3 < size_j) {
                    const float4 loaded_data = *reinterpret_cast<const float4*>(&b[row * size_j + col]);
                    smem_b[smem_row][smem_col_base + 0] = loaded_data.x;
                    smem_b[smem_row][smem_col_base + 1] = loaded_data.y;
                    smem_b[smem_row][smem_col_base + 2] = loaded_data.z;
                    smem_b[smem_row][smem_col_base + 3] = loaded_data.w;
                } else {
                    for (int elem_idx = 0; elem_idx < 4; ++elem_idx) {
                        const int32_t smem_col = smem_col_base + elem_idx;
                        const int32_t global_col = col + elem_idx;
                        smem_b[smem_row][smem_col] = (row < size_k && global_col < size_j) ? b[row * size_j + global_col] : 0.0f;
                    }
                }
            }
        }
        
        __syncthreads();
        
    
        const int32_t tile_k_start = (tile_k_begin < k_begin) ? (k_begin - tile_k_begin) : 0;
        const int32_t tile_k_stop = (tile_k_limit > k_limit) ? (k_limit - tile_k_begin) : TILE_DIM;
        
        //check if statement to allow for loop unrolling (which can't happen in the else case)
        if (tile_k_start == 0 && tile_k_stop == TILE_DIM) {

            //outer product
            #pragma unroll 8
            for (int32_t dot_idx = 0; dot_idx < TILE_DIM; ++dot_idx) {
                float a_fragment[REG_TILE_DIM];
                float b_fragment[REG_TILE_DIM];

                #pragma unroll
                for (int32_t idx = 0; idx < REG_TILE_DIM; ++idx) {
                    a_fragment[idx] = smem_a[thread_y * REG_TILE_DIM + idx][dot_idx];
                    b_fragment[idx] = smem_b[dot_idx][thread_x * REG_TILE_DIM + idx];
                }
                
                #pragma unroll
                for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {

                    #pragma unroll 

                    for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {
                        reg_accumulator[row_idx][col_idx] += a_fragment[row_idx] * b_fragment[col_idx];
                    }
                }
            }
        } 

        else {
            // Partial computation

            #pragma unroll
            for (int32_t dot_idx = tile_k_start; dot_idx < tile_k_stop; ++dot_idx) {

                float a_fragment[REG_TILE_DIM];
                float b_fragment[REG_TILE_DIM];
                #pragma unroll
                for (int32_t idx = 0; idx < REG_TILE_DIM; ++idx) {
                    a_fragment[idx] = smem_a[thread_y * REG_TILE_DIM + idx][dot_idx];
                    b_fragment[idx] = smem_b[dot_idx][thread_x * REG_TILE_DIM + idx];
                }
                
                
                #pragma unroll
                for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {

                    #pragma unroll 

                    for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {
                        reg_accumulator[row_idx][col_idx] += a_fragment[row_idx] * b_fragment[col_idx];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    //write the output    
    if ((out_row_start + TILE_DIM <= size_i) && (out_col_start + TILE_DIM <= size_j)) {
        #pragma unroll

        for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {
            #pragma unroll
            for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {
                const int32_t global_row = out_row_start + thread_y * REG_TILE_DIM + row_idx;
                const int32_t global_col = out_col_start + thread_x * REG_TILE_DIM + col_idx;
                partial_results[partition_id * (size_i * size_j) + global_row * size_j + global_col] = reg_accumulator[row_idx][col_idx];
            }
        }
    } 
    else {

        #pragma unroll
        for (int32_t row_idx = 0; row_idx < REG_TILE_DIM; ++row_idx) {
            #pragma unroll
            for (int32_t col_idx = 0; col_idx < REG_TILE_DIM; ++col_idx) {

                const int32_t global_row = out_row_start + thread_y * REG_TILE_DIM + row_idx;
                const int32_t global_col = out_col_start + thread_x * REG_TILE_DIM + col_idx;

                if (global_row < size_i && global_col < size_j) {
                    partial_results[partition_id * (size_i * size_j) + global_row * size_j + global_col] = reg_accumulator[row_idx][col_idx];
                }
            }
        }
    }
}

__global__ void reduction(
    float const *partial_results,
    float *c,
    int32_t size_i,
    int32_t size_j,
    int32_t k_splits) {
    
    int32_t elt_id = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t matrix_size = size_i * size_j;
    
    if (elt_id < matrix_size) {
        float res = 0.0f;
        
        for (int32_t split_idx = 0; split_idx < k_splits; ++split_idx) {
            res += partial_results[split_idx * matrix_size + elt_id];
        }
        c[elt_id] = res;
    }
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    constexpr int32_t TILE_SIZE = 64;

    const int32_t output_tiles = (size_i + TILE_SIZE - 1) / TILE_SIZE * (size_j + TILE_SIZE - 1) / TILE_SIZE;
    const int32_t k = (size_k + TILE_SIZE - 1) / TILE_SIZE;

    constexpr int32_t TARGET_BLOCKS = 640;

    int32_t k_splits = 1;
    if (output_tiles < TARGET_BLOCKS) {
        k_splits = (TARGET_BLOCKS + output_tiles - 1) / output_tiles;
        if (output_tiles <= 16 && k > 32) {
            k_splits = std::max(k_splits, k / 2);
        }
        k_splits = std::min(k_splits, 64);
    }

    return static_cast<size_t>(k_splits) * size_i * size_j * sizeof(float);
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    /* TODO: your CPU code here */

    constexpr int32_t TILE_SIZE = 64;
    constexpr int32_t THREADS_PER_DIM = 16;

    const int32_t output_tiles = (size_i + TILE_SIZE - 1) / TILE_SIZE * (size_j + TILE_SIZE - 1) / TILE_SIZE;
    const int32_t k = (size_k + TILE_SIZE - 1) / TILE_SIZE;
    const int32_t row = (size_i + TILE_SIZE - 1)/ TILE_SIZE;
    const int32_t col = (size_j + TILE_SIZE - 1)/ TILE_SIZE;

    constexpr int32_t TARGET_BLOCKS = 640;

    int32_t k_splits = 1;
    if (output_tiles < TARGET_BLOCKS) {
        k_splits = (TARGET_BLOCKS + output_tiles - 1) / output_tiles;
        if (output_tiles <= 16 && k > 32) {
            k_splits = std::max(k_splits, k / 2);
        }
        k_splits = std::min(k_splits, 64);
    }


    if (k_splits == 1) {
        dim3 threads_per_block(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 blocks_in_grid(col, row);
        matmul_improved::matmul_improved<<<blocks_in_grid, threads_per_block>>>(size_i, size_j, size_k, a, b, c);
    } 
    else {
        
        
        dim3 threads_per_block(THREADS_PER_DIM, THREADS_PER_DIM);
        dim3 blocks_in_grid(col, row, k_splits);
        float *partial_results = reinterpret_cast<float*>(workspace);
        int32_t elements_per_partition = (size_k + k_splits - 1) / k_splits;
        
        matmul_splitk<<<blocks_in_grid, threads_per_block>>>(size_i, size_j, size_k, a, b, partial_results, elements_per_partition);
        
        int32_t reduction_block_size = 256;
        int32_t output_elements = size_i * size_j;
        int32_t reduction_grid_size = (output_elements + reduction_block_size - 1) / reduction_block_size;
        
        reduction<<<reduction_grid_size, reduction_block_size>>>(partial_results, c, size_i, size_j, k_splits);
    }
}
        
}; // namespace matmul_improved_reduce


////////////////////////////////////////////////////////////////////////////////
// Tensor Core GPU Implementation

namespace matmul_tensor {

// Optimized tensor core kernel using m16n8k8 instruction
__global__ void __launch_bounds__(256) matmul_tensor_kernel(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {
    
    const int32_t BLOCK_M = 64;
    const int32_t BLOCK_N = 128;
    const int32_t BLOCK_K = 32;
    const int32_t WARP_M = 32;
    const int32_t WARP_N = 32;
    const int32_t MMA_M = 16;
    const int32_t MMA_N = 8;
    const int32_t MMA_K = 8;
    const int32_t warp_id = threadIdx.x / 32;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_row = warp_id / 4;
    const int32_t warp_col = warp_id % 4;
    const int32_t block_row = blockIdx.y * BLOCK_M;
    const int32_t block_col = blockIdx.x * BLOCK_N;
    const int32_t PADDING = 8;
    __shared__ float smem_a[BLOCK_K][BLOCK_M + PADDING];
    __shared__ float smem_b[BLOCK_K][BLOCK_N + PADDING];
    
    uint32_t frag_a[4];
    uint32_t frag_b[2];
    uint32_t frag_c[8][4]; 

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            frag_c[tile][i] = 0;
        }
    }
    
    for (int32_t k_tile = 0; k_tile < size_k; k_tile += BLOCK_K) {
        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int linear_idx = threadIdx.x + i * blockDim.x;
            int row_in_tile = (linear_idx * 4) / BLOCK_K;
            int col_base =(linear_idx * 4) % BLOCK_K;
            int global_row = block_row + row_in_tile;
            int global_col = k_tile + col_base;

            if (global_row < size_i && global_col + 3 < size_k) {
                float4 data = *reinterpret_cast<const float4*>(&a[global_row * size_k + global_col]);
                smem_a[col_base+0][row_in_tile] = data.x;
                smem_a[col_base+ 1][row_in_tile] = data.y;
                smem_a[col_base +2][row_in_tile] = data.z;
                smem_a[col_base+3][row_in_tile] = data.w;
            } 

            else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_row < size_i && global_col + j < size_k) {
                        smem_a[col_base + j][row_in_tile] = a[global_row * size_k + global_col + j];
                    } 

                    else {
                        smem_a[col_base + j][row_in_tile] = 0.0f;
                    }
                }
            }
        }
        
       
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear_idx= threadIdx.x + i * blockDim.x;
            int row_in_tile = (linear_idx * 4) / BLOCK_N;
            int col_base =  (linear_idx * 4) % BLOCK_N;
            int global_row = k_tile + row_in_tile;
            int global_col = block_col + col_base;

            if (global_row < size_k && global_col + 3 < size_j) {
                float4 data = *reinterpret_cast<const float4*>(&b[global_row * size_j + global_col]);
                smem_b[row_in_tile][col_base + 0] = data.x;
                smem_b[row_in_tile][col_base + 1] = data.y;
                smem_b[row_in_tile][col_base + 2] = data.z;
                smem_b[row_in_tile][col_base + 3] = data.w;
            } 
            
            else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_row < size_k && global_col + j < size_j) {
                        smem_b[row_in_tile][col_base + j] = b[global_row * size_j + global_col + j];
                    } 
                    
                    else {
                        smem_b[row_in_tile][col_base+ j] = 0.0f;
                    }
                }
            }
        }
        
        __syncthreads();
        
        
        #pragma unroll
        for (int k_step = 0; k_step < BLOCK_K; k_step += MMA_K) {
            
            #pragma unroll
            for (int mma_m = 0; mma_m < 2; ++mma_m) {
                #pragma unroll
                for (int mma_n = 0; mma_n < 4; ++mma_n) {
                    int a_row = warp_row * WARP_M +mma_m * MMA_M;
                    int a_col = k_step;
                    int b_row = k_step;
                    int b_col = warp_col* WARP_N + mma_n * MMA_N;
                    
                    
                    int a_thread_row = lane_id / 4;
                    int a_thread_col = lane_id % 4;
                    
                    frag_a[0] = __float_as_uint(smem_a[a_col + a_thread_col][a_row + a_thread_row]);
                    frag_a[1] = __float_as_uint(smem_a[a_col + a_thread_col][a_row + a_thread_row + 8]);
                    frag_a[2] = __float_as_uint(smem_a[a_col + a_thread_col + 4][a_row + a_thread_row]);
                    frag_a[3] = __float_as_uint(smem_a[a_col + a_thread_col + 4][a_row + a_thread_row + 8]);
                    
                    
                    int b_thread_col = lane_id / 4;
                    int b_thread_row = lane_id % 4;
                    
                    frag_b[0] = __float_as_uint(smem_b[b_row + b_thread_row][b_col + b_thread_col]);
                    frag_b[1] = __float_as_uint(smem_b[b_row + b_thread_row + 4][b_col + b_thread_col]);
                    
                    int tile_idx = mma_m * 4 + mma_n;
                    uint32_t *frag_c_ptr = frag_c[tile_idx];
                    
                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                        : "=r"(frag_c_ptr[0]), "=r"(frag_c_ptr[1]), "=r"(frag_c_ptr[2]), "=r"(frag_c_ptr[3])
                        :"r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                          "r"(frag_b[0]), "r"(frag_b[1]),
                          "r"(frag_c_ptr[0]), "r"(frag_c_ptr[1]), "r"(frag_c_ptr[2]), "r"(frag_c_ptr[3])
                      );
                }
            }
                }
                
                __syncthreads();
            }
            
            auto write_tile = [&](uint32_t *frag, int mma_m, int mma_n) {
                int out_row_base = block_row + warp_row * WARP_M + mma_m * MMA_M;
                int out_col_base = block_col + warp_col * WARP_N + mma_n * MMA_N;
                int c_thread_row = lane_id / 4;
                int c_thread_col =lane_id % 4;
                int row0 = out_row_base + c_thread_row;
                int col0 = out_col_base + c_thread_col * 2;
                if (row0 < size_i &&col0 < size_j) {
                    c[row0 * size_j + col0] = __uint_as_float(frag[0]);
                }
                
                int col1 = out_col_base + c_thread_col * 2 + 1;
                if (row0 < size_i && col1 < size_j) {
                    c[row0 * size_j + col1] = __uint_as_float(frag[1]);
                }
                
                int row2 = out_row_base + c_thread_row + 8;
                if (row2 < size_i && col0 < size_j) {
                    c[row2 * size_j + col0] = __uint_as_float(frag[2]);
                }
                
                if (row2 < size_i && col1 < size_j) {
                    c[row2 * size_j + col1] = __uint_as_float(frag[3]);
                }
            };
            
            #pragma unroll
            for (int mma_m = 0; mma_m < 2;++mma_m) {
                #pragma unroll
                for (int mma_n = 0; mma_n < 4; ++mma_n) {
                    write_tile(frag_c[mma_m * 4 + mma_n], mma_m, mma_n);
                }
            }
        }
        
__global__ void __launch_bounds__(256) matmul_tensor_splitk(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *partial_results,
    int32_t k_per_split,
    int32_t k_start_offset) {

    const int32_t BLOCK_M = 64;
    const int32_t BLOCK_N = 128;
    const int32_t BLOCK_K = 32;
    const int32_t WARP_M = 32;
    const int32_t WARP_N = 32;
    const int32_t MMA_M = 16;
    const int32_t MMA_N = 8;
    const int32_t MMA_K = 8;
    const int32_t split_id = blockIdx.z;
    const int32_t k_start = k_start_offset + split_id * k_per_split;
    const int32_t k_end = min(k_start + k_per_split, size_k);
    const int32_t warp_id = threadIdx.x / 32;
    const int32_t lane_id = threadIdx.x % 32;
    const int32_t warp_row = warp_id / 4;
    const int32_t warp_col = warp_id % 4;
    const int32_t block_row = blockIdx.y * BLOCK_M;
    const int32_t block_col = blockIdx.x * BLOCK_N;

    const int32_t PADDING = 8;
    __shared__ float smem_a[BLOCK_K][BLOCK_M + PADDING];
    __shared__ float smem_b[BLOCK_K][BLOCK_N + PADDING];

    uint32_t frag_a[4];
    uint32_t frag_b[2];
    uint32_t frag_c[8][4];

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            frag_c[tile][i] = 0;
        }
    }

    for (int32_t k_tile = k_start; k_tile < k_end; k_tile += BLOCK_K) {
        int32_t k_remaining = min(BLOCK_K, k_end - k_tile);

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int linear_idx = threadIdx.x + i * blockDim.x;
            int row_in_tile = (linear_idx * 4)/ BLOCK_K;
            int col_base = (linear_idx * 4) % BLOCK_K;
            int global_row = block_row + row_in_tile;
            int global_col = k_tile + col_base;

            if (global_row < size_i && global_col + 3 < size_k && col_base + 3 < k_remaining) {
                float4 data = *reinterpret_cast<const float4*>(&a[global_row * size_k + global_col]);
                smem_a[col_base + 0][row_in_tile] = data.x;
                smem_a[col_base + 1][row_in_tile] = data.y;
                smem_a[col_base +2][row_in_tile] = data.z;
                smem_a[col_base+ 3][row_in_tile] = data.w;
            } 
            
            else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_row < size_i && global_col + j < size_k && col_base + j < k_remaining) {
                        smem_a[col_base + j][row_in_tile] = a[global_row * size_k + global_col + j];
                    } 
                    
                    else {
                        smem_a[col_base + j][row_in_tile] = 0.0f;
                    }
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear_idx = threadIdx.x + i * blockDim.x;
            int row_in_tile = (linear_idx * 4) / BLOCK_N;
            int col_base = (linear_idx * 4) % BLOCK_N;
            int global_row = k_tile + row_in_tile;
            int global_col = block_col + col_base;

            if (global_row < size_k && global_col + 3 < size_j && row_in_tile < k_remaining) {
                float4 data = *reinterpret_cast<const float4*>(&b[global_row * size_j + global_col]);
                smem_b[row_in_tile][col_base + 0] = data.x;
                smem_b[row_in_tile][col_base + 1] = data.y;
                smem_b[row_in_tile][col_base + 2] = data.z;
                smem_b[row_in_tile][col_base + 3] = data.w;
            } 
            
            else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (global_row < size_k && global_col + j < size_j && row_in_tile < k_remaining) {
                        smem_b[row_in_tile][col_base + j] = b[global_row * size_j + global_col + j];
                    } 
                    
                    else {
                        smem_b[row_in_tile][col_base + j] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        int32_t k_steps = (k_remaining + MMA_K - 1) / MMA_K;
        for (int k_step_idx = 0; k_step_idx < k_steps; ++k_step_idx) {
            int k_step = k_step_idx * MMA_K;

            #pragma unroll
            for (int mma_m = 0; mma_m < 2; ++mma_m) {
                #pragma unroll
                for (int mma_n = 0; mma_n < 4; ++mma_n) {
                    int a_row = warp_row * WARP_M + mma_m * MMA_M;
                    int a_col = k_step;
                    int b_row =k_step;
                    int b_col = warp_col * WARP_N + mma_n * MMA_N;

                    int a_thread_row = lane_id / 4;
                    int a_thread_col = lane_id % 4;

                    frag_a[0] = __float_as_uint(smem_a[a_col + a_thread_col][a_row + a_thread_row]);
                    frag_a[1] = __float_as_uint(smem_a[a_col + a_thread_col][a_row + a_thread_row + 8]);
                    frag_a[2] = __float_as_uint(smem_a[a_col + a_thread_col + 4][a_row + a_thread_row]);
                    frag_a[3] = __float_as_uint(smem_a[a_col + a_thread_col + 4][a_row + a_thread_row + 8]);

                    int b_thread_col = lane_id / 4;
                    int b_thread_row = lane_id % 4;

                    frag_b[0] = __float_as_uint(smem_b[b_row + b_thread_row][b_col + b_thread_col]);
                    frag_b[1] = __float_as_uint(smem_b[b_row + b_thread_row + 4][b_col + b_thread_col]);

                    int tile_idx = mma_m * 4 + mma_n;
                    uint32_t *frag_c_ptr = frag_c[tile_idx];

                    asm volatile(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                        : "=r"(frag_c_ptr[0]), "=r"(frag_c_ptr[1]), "=r"(frag_c_ptr[2]), "=r"(frag_c_ptr[3])
                        : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                            "r"(frag_b[0]), "r"(frag_b[1]),
                            "r"(frag_c_ptr[0]), "r"(frag_c_ptr[1]), "r"(frag_c_ptr[2]), "r"(frag_c_ptr[3])
                    );
                }
            }
        }

        __syncthreads();
    }

    auto write_tile = [&](uint32_t *frag, int mma_m, int mma_n) {
        int out_row_base = block_row + warp_row * WARP_M + mma_m * MMA_M;
        int out_col_base = block_col + warp_col * WARP_N + mma_n * MMA_N;

        int c_thread_row = lane_id / 4;
        int c_thread_col = lane_id % 4;

        int row0 = out_row_base + c_thread_row;
        int col0 = out_col_base + c_thread_col * 2;
        int col1 = out_col_base + c_thread_col * 2 + 1;
        int row2 = out_row_base + c_thread_row + 8;

        int offset = split_id * size_i * size_j;

        if (row0 < size_i && col0 < size_j) {
            partial_results[offset + row0 * size_j + col0] = __uint_as_float(frag[0]);
        }
        if (row0 < size_i && col1 < size_j) {
            partial_results[offset + row0 * size_j + col1] = __uint_as_float(frag[1]);
        }
        if (row2 < size_i && col0 < size_j) {
            partial_results[offset + row2 * size_j + col0] = __uint_as_float(frag[2]);
        }
        if (row2 < size_i && col1 < size_j) {
            partial_results[offset + row2 * size_j + col1] = __uint_as_float(frag[3]);
        }
    };

    #pragma unroll
    for (int mma_m = 0; mma_m < 2; ++mma_m) {
        #pragma unroll
        for (int mma_n = 0; mma_n < 4; ++mma_n) {
            write_tile(frag_c[mma_m * 4 + mma_n], mma_m, mma_n);
        }
    }
}

__global__ void reduce_splits(
    float const *partial_results,
    float *c,
    int32_t size_i,
    int32_t size_j,
    int32_t num_splits) {
    
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_elements = size_i * size_j;
    
    if (idx < total_elements) {
        float sum = 0.0f;
        for (int32_t split = 0; split < num_splits; ++split) {
            sum += partial_results[split * total_elements + idx];
        }
        c[idx] = sum;
    }
}
        
size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    constexpr int32_t BLOCK_M = 64;
    constexpr int32_t BLOCK_N = 128;

    int32_t grid_m = (size_i + BLOCK_M - 1) / BLOCK_M;
    int32_t grid_n = (size_j + BLOCK_N - 1) / BLOCK_N;
    int32_t total_blocks = grid_m * grid_n;

    int32_t num_splits = 0;

    if (size_i == 256) {
        num_splits = 6;
    } 
    else if (size_i == 512) {
        num_splits = 3;
    } 
    else if (total_blocks < 100 && size_k >= 1024) {
        num_splits = min(8, size_k / 512);
    }

    if (num_splits > 1) {
        return static_cast<size_t>(num_splits) * size_i * size_j * sizeof(float);
    }

    return 0;
}

void launch_matmul_tensor(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c,       /* pointer to GPU memory */
    void *workspace /* pointer to GPU memory */
) {
    if (size_i < 256 || size_j < 256) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
        return;
    }
    
    constexpr int32_t BLOCK_M = 64;
    constexpr int32_t BLOCK_N = 128;
    
    dim3 block(256);  
    dim3 grid((size_j + BLOCK_N - 1) / BLOCK_N,
              (size_i + BLOCK_M - 1) / BLOCK_M);
    
    size_t workspace_size = get_workspace_size(size_i, size_j, size_k);
    
    if (workspace_size > 0) {
        int32_t num_splits = workspace_size / (size_i * size_j * sizeof(float));
        int32_t k_per_split = (size_k + num_splits - 1) / num_splits;
        
        dim3 split_grid(grid.x, grid.y, num_splits);
        float *partial_results = static_cast<float*>(workspace);
        
        matmul_tensor_splitk<<<split_grid, block>>>(
            size_i, size_j, size_k, a, b, partial_results, k_per_split, 0);
        
        int32_t reduce_threads = 256;
        int32_t reduce_blocks = (size_i * size_j + reduce_threads - 1) / reduce_threads;
        reduce_splits<<<reduce_blocks, reduce_threads>>>(
            partial_results, c, size_i, size_j, num_splits);
    } 
    
    else {
        matmul_tensor_kernel<<<grid, block>>>(size_i, size_j, size_k, a, b, c);
    }
}

}; // namespace matmul_tensor

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
    CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        a_gpu,
        a.data(),
        size_i * size_k * sizeof(float),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        b_gpu,
        b.data(),
        size_k * size_j * sizeof(float),
        cudaMemcpyHostToDevice));

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

    std::vector<float> c_out_host(size_i * size_j);
    CUDA_CHECK(cudaMemcpy(
        c_out_host.data(),
        c_gpu,
        size_i * size_j * sizeof(float),
        cudaMemcpyDeviceToHost));

    double mse = 0.0;
    double ref_mean_square = 0.0;
    for (int32_t i = 0; i < size_i; ++i) {
        for (int32_t j = 0; j < size_j; ++j) {
            float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
            mse += diff * diff;
            ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
        }
    }
    mse /= size_i * size_j;
    ref_mean_square /= size_i * size_j;
    float rmse = std::sqrt(mse);
    float rel_rmse = rmse / std::sqrt(ref_mean_square);

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-3) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 200.0;
        double elapsed_ms = benchmark_ms(
            target_time_ms,
            1,
            [&]() {
                if (workspace_size > 0) {
                    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                }
                CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
            },
            [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
            });

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
    CUDA_CHECK(cudaFree(flush_gpu));
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_5_BASELINE_IMPL

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

#endif

struct MatmulTensor {
    constexpr static char const *name = "matmul_tensor";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_tensor::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_tensor::launch_matmul_tensor(size_i, size_j, size_k, a, b, c, workspace);
    }
};

BenchmarkResults get_cublas_fma_results() {
    // Hard-coded data collected on A4000 GPU
    return BenchmarkResults{
        "cublas_fma",
        {
            {{3072, 3072, 3072}, 3.152},
            {{2048, 3072, 3072}, 2.174},
            {{1024, 3072, 3072}, 1.090},
            {{512, 3072, 3072}, 0.559},
            {{256, 3072, 3072}, 0.356},
            {{128, 3072, 3072}, 0.256},
            {{64, 3072, 3072}, 0.194},
            {{32, 3072, 3072}, 0.181},
            {{16, 3072, 3072}, 0.181},
        }};
}

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_5_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulTensor>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

void print_speedup(
    std::vector<BenchmarkConfig> const &configs,
    BenchmarkResults const &first,
    BenchmarkResults const &second) {
    printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
    printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
    printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
        auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
        auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
        if (it_first != first.elapsed_ms.end() && it_second != second.elapsed_ms.end()) {
            printf("  %6.02fx", it_first->second / it_second->second);
        } else {
            printf("  %7s", "-");
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    std::string test_data_dir = ".";


    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {2048, 3072, 3072},
        {1024, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            print_speedup(configs, results.at(i), results.at(j));
        }
    }

    printf("\n-----------------------------------------------------------\n");
    printf("---- Comparison to non-tensor-core cuBLAS performance: ----\n");
    printf("-----------------------------------------------------------\n");

    print_speedup(configs, get_cublas_fma_results(), results.at(results.size() - 1));

    write_json_results("out/results.json", results);

    return 0;
}
