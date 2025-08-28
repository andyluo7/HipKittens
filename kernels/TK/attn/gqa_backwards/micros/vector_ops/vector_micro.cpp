#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int b = 1;
constexpr int h = 1;
constexpr int n = 32;
constexpr int d = 32;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in_a, in_b; 
    gl<float, -1, -1, -1, -1> out, in_accum;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY-2048; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {

    rt_bf<n, d, row_l> a_tile;
    rt_bf<n, d, row_l> b_tile;
    rt_fl<n, n, accum_col_l> tile_accum;
    load(a_tile, g.in_a, {0, 0, 0, 0});
    load(b_tile, g.in_b, {0, 0, 0, 0});
    load(tile_accum, g.in_accum, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();


    mma_ABt(tile_accum, a_tile, b_tile, tile_accum);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();

    
    store(g.out, tile_accum, {0, 0, 0, 0});
    
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __syncthreads();
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in_a, &micro_globals::in_b, &micro_globals::out, &micro_globals::in_accum);
}

