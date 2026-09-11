# PA Decode Kernel Regression: Triton 3.7 vs 3.8

## Shape
B=4, Q=1, QH=8, KVH=1, D=256, block=16, context=4096, FP8 per-token, ps=True

## Kernel
`paged_attention_decode_sliding_window_head_1`

## Performance
| Metric | Triton 3.7 | Triton 3.8 | Ratio |
|--------|-----------|-----------|-------|
| Kernel time | 13.48 us | 33.88 us | 2.51x |

## Files
- `triton_3.7_cache/` — Triton 3.7 compiled cache (.ttgir, .llir, .amdgcn, .hsaco)
- `triton_3.8_cache/` — Triton 3.8 compiled cache (.ttgir, .llir, .amdgcn, .hsaco)
- `llvm_ir_dump_triton3.7.log` — LLVM O3 pass-by-pass IR dump (LLVM_IR_ENABLE_DUMP=1)
- `llvm_ir_dump_triton3.8.log` — LLVM O3 pass-by-pass IR dump (LLVM_IR_ENABLE_DUMP=1)

## Wheel packages
- 3.7: `pip install triton==3.7.0+amd.rocm7.2.0.git89002410 --index-url https://pypi.amd.com/triton/release/rocm-7.2.0/simple/`
- 3.8: `pip install triton==3.8.0+amd.rocm7.2.0.git111ff227 --index-url https://pypi.amd.com/triton/_release/rocm-7.2.0/simple/`
