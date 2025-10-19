
## Baseline kernels

This README describes the baseline kernels we use from third party libraries at the time of this work (October 2025). We benchmarked all kernels using 500 warmup and 100 repeat iterations. 

### Setting up composable kernel

Baselines were collected using this process:
```bash
[~] git clone https://github.com/rocm/composable_kernel
[~] cd composable_kernel
[~/composable_kernel] mkdir build && cd build
[~/composable_kernel/build] ../script/cmake-ck-dev.sh .. gfx950 -G Ninja
[~/composable_kernel/build] ninja tile_example_gemm_basic
```

Just in case, here is a working commit at the time of this work [commit](https://github.com/ROCm/composable_kernel/tree/d88ea05c844cd159a14213b73a5818a43c5b79e6).

For attention, we ran the above for ```ninja tile_example_fmha_fwd``` and ```ninja tile_example_fmha_bwd```. Then we benchmarked with the following commands. We only found a ```ck_tile``` example in the repository at the time of this work (October 2025):

Forwards:
```bash
# non-causal forwards mha
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal forwards mha
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1

# non-causal forwards gqa
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal forwards gqa
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_fwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1
```

Backwards:
```bash
# non-causal mha
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal mha
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=16 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1

# non-causal gqa
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -warmup=500 -repeat=100 -kname=1

# causal gqa
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=1024 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=2048 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=4096 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=8192 -mask=1 -warmup=500 -repeat=100 -kname=1
./bin/tile_example_fmha_bwd -prec=bf16 -b=16 -h=64 -h_k=8 -d=128 -s=16384 -mask=1 -warmup=500 -repeat=100 -kname=1
```

### Triton baselines

Attention baselines for triton are taken as the best performance out of:
- [ROCm triton perf-kernels](https://github.com/ROCm/triton/tree/76076e1d7d16a988a61a66264845990acd1244ab/python/perf-kernels) ```flash-attention.py```
- [ROCm triton tutorials](https://github.com/ROCm/triton/tree/76076e1d7d16a988a61a66264845990acd1244ab/python/tutorials) ```06-fused-attention.py```

We can directly run the files using python. 


### HipblasLT baselines





