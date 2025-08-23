#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define NUM_WORKERS (1) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

constexpr int ATTN_B = 1;
constexpr int ATTN_H = 1;
constexpr int N = 64;
constexpr int head_dim = 64;                   
constexpr float rope_embd_fraction = 1.0f;

// Make these constexpr so they can be used as template parameters
constexpr int rope_dim = static_cast<int>(rope_embd_fraction * head_dim);
constexpr int half_rope_dim = (rope_dim / 2);
constexpr int excess_dim = head_dim - rope_dim; 
constexpr int BLOCK_SIZE = 32;

using namespace kittens;

#define tile_1xFULL_ROPE_D st<bf16, BLOCK_SIZE, rope_dim>
#define tile_1xHALF_ROPE_D st<bf16, BLOCK_SIZE, half_rope_dim>
#define tile_1xEXCESS_ROPE_D st<bf16, BLOCK_SIZE, excess_dim>

#define reg_tile_1xFULL_ROPE_D rt<bf16, BLOCK_SIZE, rope_dim>
#define reg_tile_1xHALF_ROPE_D rt<bf16, BLOCK_SIZE, half_rope_dim>
#define reg_tile_1xEXCESS_ROPE_D rt<bf16, BLOCK_SIZE, excess_dim>

template<int _d_model> struct rotary_globals {
    static constexpr int d_model = _d_model;

    // global descriptors
    using x_gl = gl<bf16, -1, -1, -1, -1>;
    using o_gl = gl<bf16, -1, -1, -1, -1>;
    using sin_gl = gl<bf16, -1, -1, -1, -1>;
    using cos_gl = gl<bf16, -1, -1, -1, -1>;

    // global pointers
    x_gl x;
    o_gl o;
    sin_gl sin;
    cos_gl cos;

    dim3 grid() { return dim3(ATTN_B, ATTN_H); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY / 2; }
};

template<int D>
__device__ inline void apply_rotary_embedding(reg_tile_1xFULL_ROPE_D &x_reg,
                                              const reg_tile_1xHALF_ROPE_D &cos_reg,
                                              const reg_tile_1xHALF_ROPE_D &sin_reg) {
    reg_tile_1xHALF_ROPE_D x1, x2, temp1, temp2;
    
    int half_dim_tiles = half_rope_dim / BLOCK_SIZE;  // Should be 1 for D=64

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    int condition = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);
    if (condition) {
        printf("half_dim_tiles: %d\n", half_dim_tiles);
        printf("reg_tile_1xHALF_ROPE_D::packed_per_thread: %d\n", reg_tile_1xHALF_ROPE_D::packed_per_thread);
    }
    
    // Copy first half and second half
    for(int i = 0; i < half_dim_tiles; i++) {
        #pragma unroll
        for(int j = 0; j < reg_tile_1xHALF_ROPE_D::packed_per_thread; j++){
            x1.tiles[0][i].data[j] = x_reg.tiles[0][i].data[j];                    // First half: elements 0-31
            x2.tiles[0][i].data[j] = x_reg.tiles[0][i + half_dim_tiles].data[j];   // Second half: elements 32-63
        }
    }
    
    // Apply rotary embedding transformation:
    // new_x1 = x1 * cos - x2 * sin  
    // new_x2 = x2 * cos + x1 * sin
    
    // Compute x1 * cos
    mul(temp1, x1, cos_reg);
    
    // Compute x2 * sin  
    mul(temp2, x2, sin_reg);
    
    // new_x1 = x1 * cos - x2 * sin
    sub(temp1, temp1, temp2);
    
    // Compute x2 * cos
    mul(temp2, x2, cos_reg);
    
    // Compute x1 * sin
    mul(x1, x1, sin_reg);
    
    // new_x2 = x2 * cos + x1 * sin
    add(temp2, temp2, x1);
    
    // Write results back to x_reg
    for(int i = 0; i < half_dim_tiles; i++) {
        #pragma unroll
        for(int j = 0; j < reg_tile_1xHALF_ROPE_D::packed_per_thread; j++) {
            x_reg.tiles[0][i].data[j] = temp1.tiles[0][i].data[j];                    // First half
            x_reg.tiles[0][i + half_dim_tiles].data[j] = temp2.tiles[0][i].data[j];   // Second half  
        }
    }
}

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void _fused_rotary(const rotary_globals<D> g) {
    auto warpid = kittens::warpid();
    auto lane = kittens::laneid();

    const int b = blockIdx.x;
    const int h = blockIdx.y;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    tile_1xFULL_ROPE_D (&x_s) = al.allocate<tile_1xFULL_ROPE_D>();    
    tile_1xHALF_ROPE_D (&cos_s) = al.allocate<tile_1xHALF_ROPE_D>(); 
    tile_1xHALF_ROPE_D (&sin_s) = al.allocate<tile_1xHALF_ROPE_D>(); 

    int n_blocks = N / (NUM_WORKERS * BLOCK_SIZE);
    for (int block = 0; block < n_blocks; block++) {

        // Load data from global memory to shared memory
        load(x_s, g.x, {b, h, block, 0});
        load(cos_s, g.cos, {0, 0, block, 0});  // cos and sin are shared across batch/head
        load(sin_s, g.sin, {0, 0, block, 0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        // Load from shared memory to registers
        reg_tile_1xFULL_ROPE_D x_reg;
        reg_tile_1xHALF_ROPE_D cos_reg, sin_reg;
        
        load(x_reg, x_s);
        load(cos_reg, cos_s);
        load(sin_reg, sin_s);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __syncthreads();

        // Apply rotary embedding
        apply_rotary_embedding<D>(x_reg, cos_reg, sin_reg);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        // Store result back to global memory
        store(g.o, x_reg, {b, h, block, 0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
    }
}

template<int D>
void dispatch_rotary(rotary_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)_fused_rotary<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    _fused_rotary<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_rotary<head_dim>>(m, "dispatch_rotary", 
        &rotary_globals<head_dim>::x, 
        &rotary_globals<head_dim>::o, 
        &rotary_globals<head_dim>::sin, 
        &rotary_globals<head_dim>::cos
    );
}