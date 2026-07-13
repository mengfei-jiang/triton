"""
End-to-end test: verify that enable_sgpr_kernarg_preload per-kernel option
controls inreg marking AND the kernel still produces correct results in both modes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def vecadd_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def get_inreg_count(kernel, *args, **kwargs):
    h = kernel.warmup(*args, **kwargs)
    llir = h.asm["llir"]
    for line in llir.split("\n"):
        if line.strip().startswith("define") and "vecadd_kernel" in line:
            return line.count("inreg")
    return 0


def main():
    N = 1024
    BLOCK = 256
    device = "cuda"

    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.randn(N, device=device, dtype=torch.float32)
    ref = x + y

    grid = lambda meta: ((N + meta["BLOCK"] - 1) // meta["BLOCK"], )

    # --- preload ON (default) ---
    out_on = torch.empty_like(x)
    vecadd_kernel[grid](x, y, out_on, N, BLOCK=BLOCK, enable_sgpr_kernarg_preload=True)
    n_inreg_on = get_inreg_count(
        vecadd_kernel, x, y, out_on, N,
        BLOCK=BLOCK, grid=(N // BLOCK, ), enable_sgpr_kernarg_preload=True,
    )

    # --- preload OFF ---
    out_off = torch.empty_like(x)
    vecadd_kernel[grid](x, y, out_off, N, BLOCK=BLOCK, enable_sgpr_kernarg_preload=False)
    n_inreg_off = get_inreg_count(
        vecadd_kernel, x, y, out_off, N,
        BLOCK=BLOCK, grid=(N // BLOCK, ), enable_sgpr_kernarg_preload=False,
    )

    # --- check correctness ---
    assert torch.allclose(out_on, ref), "Wrong result with preload ON"
    assert torch.allclose(out_off, ref), "Wrong result with preload OFF"

    # --- check IR difference ---
    print(f"enable_sgpr_kernarg_preload=True  -> inreg count: {n_inreg_on}")
    print(f"enable_sgpr_kernarg_preload=False -> inreg count: {n_inreg_off}")
    assert n_inreg_on > 0, "Expected inreg when preload is ON"
    assert n_inreg_off == 0, f"Expected 0 inreg when preload is OFF, got {n_inreg_off}"

    print("\nPASS: per-kernel preload control works — both modes produce correct results.")


if __name__ == "__main__":
    main()
