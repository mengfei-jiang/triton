# Triton Gluon Extension for AMDGPU

This repo provides some extension to current Gluon to give users more control without breaking the tile level programming model.

## Triton prepare

1. git cherry-pick tip commits from [gluon_ext](https://github.com/ROCm/triton/tree/gluon_ext), also need to rebase it, so we can make gluon_ext clean.
2. build triton from source. (e.g. `pip install -e .`)

common debug:
cherry-pick should be conflict free since this is add-on.
if your local triton is out-of-date for long time, you may get compile error.
e.g. comment out the warp_id implementation in `python/triton/experimental/gluon/language/amd/cdna3/__init__.py`
if your code hasn't contained the implementation of ttg.warp_id

## Kernel Authoring

1. import gluon

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

2. using the amd extensions, mostly gl.amd.cdna3/4.xxx

### extensions

1. instruction scheduling related.
  Since none of these are fully respected, we cover it here.
  - sched_barrier: separate scheduling regions so that instrs can not pass it, generally respected by llvm compiler, but not guaranteed in all cases.

    sched_barrier(mask) controls which instruction types may be reordered across this point by the llvm scheduler.
    Two sched_barrier(0x0) calls define a hard scheduling region: no instructions may move in or out.

    mask values (combinable via bitwise OR):
  ```
  0x000 = NONE       → hard fence, nothing crosses
  0x001 = ALU        → non-memory, non-side-effect
  0x002 = VALU       → vector ALU
  0x004 = SALU       → scalar ALU
  0x008 = MFMA/WMMA  → matrix multiply-accumulate
  0x010 = ALL VMEM   → all vector memory (read + write)
  0x020 = VMEM read  → vector memory reads only
  0x040 = VMEM write → vector memory writes only
  0x080 = ALL DS     → all LDS operations (read + write)
  0x100 = DS read    → LDS reads only
  0x200 = DS write   → LDS writes only
  ```

  - sched_group_barrier: specify certain order in the scheduling region. e.g 1 mma followed by 4 ds_read. may not be respected by llvm compiler.

      sched_group_barrier(mask, count, group_id) tells llvm to schedule `count` instructions matching `mask` before crossing this point.
      For the `mask`, we can take sched_barrier as a reference. Instructions are selected bottom-up from the barrier's position.
      The third parameter, `group_id`, identifies which other sched_group_barriers should be synchronized with.

  - iglp_opt: setting scheduling strategy, not effective all the time
2. control related
  - s_barrier: Synchronize threads/workitems within a threadgroup/workgroup
  - s_set_prio: set the priority of warp/wave
  - warp_id: warp/wave id inside a threadgroup/workgroup

For more details, please refer to `python/triton/experimental/gluon/language/amd/cdna3/__init__.py`

### examples

`python bench_matmul.py` will run the matmul examples including triton, gluon, gluon_ext.

- `matmul_triton.py` is the triton version.
- `matmul_gluon_gfx950_256x256x32_3stage.py` and `matmul_gluon_gfx950_256x256x64_2stage.py` are gluon version.
- `matmul_gluon_gfx950_256x256x32_3stage_scheduling.py` and `matmul_gluon_gfx950_256x256x64_2stage_scheduling.py` are gluon_ext version, manually control instruction scheduling.
- `matmul_gluon_gfx950_256x256x32_3stage_pingpong.py` and `matmul_gluon_gfx950_256x256x64_2stage_pingpong.py` are gluon_ext version, implementing a block ping-pong strategy.
