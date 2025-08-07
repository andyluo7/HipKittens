#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define SIZE 64

using _gl_tile = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    _gl_tile input;
    _gl_tile output;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; } 
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    rt_fl<SIZE, SIZE> tile_fl;
    rt_f6<SIZE, SIZE> tile_f6;
    rt_fl<SIZE, SIZE, accum_l> tile_fl_accum;
    zero(tile_fl_accum);
    one(tile_f6);
    mma_ABt(tile_fl_accum, tile_f6, tile_f6, tile_fl_accum);
    __syncthreads();
    // copy(tile_fl, tile_f6);
    store(g.output, tile_fl_accum, {0, 0, 0, 0});
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module"; 
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::input, &micro_globals::output);
}
