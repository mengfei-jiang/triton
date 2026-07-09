"""
Test that HIPOptions.enable_sgpr_kernarg_preload controls whether kernel
arguments are marked 'inreg' in the LLVM IR (compile-only, no device needed).
"""
import triton
import triton.language as tl


@triton.jit
def simple_add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    tl.store(out_ptr + offs, x + y)


def compile_and_get_llir(enable_preload):
    """Compile simple_add_kernel with the given preload setting, return LLVM IR."""
    from triton.runtime import driver
    target = driver.active.get_current_target()

    options = {"enable_sgpr_kernarg_preload": enable_preload}

    compiled = triton.compile(
        triton.compiler.ASTSource(
            fn=simple_add_kernel,
            signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32"},
            constexprs={"N": 128},
        ),
        options=options,
    )
    return compiled.asm["llir"]


def count_inreg(llir):
    """Count 'inreg' occurrences in the kernel define line."""
    for line in llir.split("\n"):
        if line.strip().startswith("define") and "simple_add_kernel" in line:
            return line.count("inreg")
    return 0


def main():
    # --- Case 1: preload enabled (default) ---
    llir_on = compile_and_get_llir(enable_preload=True)
    n_inreg_on = count_inreg(llir_on)

    # --- Case 2: preload disabled ---
    llir_off = compile_and_get_llir(enable_preload=False)
    n_inreg_off = count_inreg(llir_off)

    print(f"enable_sgpr_kernarg_preload=True  -> inreg count: {n_inreg_on}")
    print(f"enable_sgpr_kernarg_preload=False -> inreg count: {n_inreg_off}")

    # When preload is on, kernel args should be marked inreg
    assert n_inreg_on > 0, "Expected inreg attrs when preload is enabled"
    # When preload is off, no inreg should appear
    assert n_inreg_off == 0, f"Expected 0 inreg attrs when preload is disabled, got {n_inreg_off}"

    print("\nPASS: enable_sgpr_kernarg_preload correctly controls inreg marking.")


if __name__ == "__main__":
    main()
