import pytest
import torch

import triton
import triton.language as tl
from triton.language.core import PropagateNan

# ===-----------------------------------------------------------------------===#
# Max/min Utilities
# ===-----------------------------------------------------------------------===#

_MAX_PROPAGATE_NAN_ALL = tl.constexpr(PropagateNan.ALL)


@triton.jit
def elementwise_max_prop_nan(a, b):
    return tl.maximum(a, b, propagate_nan=_MAX_PROPAGATE_NAN_ALL)


@triton.jit
def reduce_max_prop_nan(input, axis=None, keep_dims=False):
    """Returns the max of the input tensor along the provided axis.

    We need this customized impl rather than tl.max in order to control NaN
    behavior. Ignoring NaN would incur extra overhead on AMD GPUs."""
    return tl.reduce(input, axis, elementwise_max_prop_nan, keep_dims=keep_dims)


@triton.jit
def elementwise_min_prop_nan(a, b):
    return tl.minimum(a, b, propagate_nan=_MAX_PROPAGATE_NAN_ALL)


@triton.jit
def reduce_min_prop_nan(input, axis=None, keep_dims=False):
    """Returns the min of the input tensor along the provided axis.

    We need this customized impl rather than tl.min in order to control NaN
    behavior. Ignoring NaN would incur extra overhead on AMD GPUs."""
    return tl.reduce(input, axis, elementwise_min_prop_nan, keep_dims=keep_dims)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


@triton.jit
def reduce_max_kernel(out_ptr, in_ptr, M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    x = tl.load(in_ptr + rows[:, None] * N + cols[None, :])
    m = reduce_max_prop_nan(x, axis=1)
    tl.store(out_ptr + rows, m)


@triton.jit
def reduce_min_kernel(out_ptr, in_ptr, M: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    x = tl.load(in_ptr + rows[:, None] * N + cols[None, :])
    m = reduce_min_prop_nan(x, axis=1)
    tl.store(out_ptr + rows, m)


def _run_reduce_max(x, M, N):
    out = torch.empty(M, dtype=torch.float32, device=x.device)
    reduce_max_kernel[(1, )](out, x, M, N)
    return out


def _run_reduce_min(x, M, N):
    out = torch.empty(M, dtype=torch.float32, device=x.device)
    reduce_min_kernel[(1, )](out, x, M, N)
    return out


@pytest.mark.parametrize("op", ["max", "min"])
@pytest.mark.parametrize("M,N", [(8, 64)])
def test_reduce_prop_nan(op, M, N):
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")
    x[1, 10] = float("nan")
    x[5, 30] = float("nan")

    if op == "max":
        out = _run_reduce_max(x, M, N)
        expected = x.max(dim=1)[0]
    else:
        out = _run_reduce_min(x, M, N)
        expected = x.min(dim=1)[0]

    # Rows with NaN should produce NaN
    assert torch.isnan(out[1]).item(), f"Row 1 has NaN, {op} should be NaN"
    assert torch.isnan(out[5]).item(), f"Row 5 has NaN, {op} should be NaN"

    # Clean rows should match torch result
    clean_rows = [0, 2, 3, 4, 6, 7]
    for r in clean_rows:
        torch.testing.assert_close(out[r], expected[r])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8, help="number of rows")
    parser.add_argument("-N", type=int, default=64, help="number of columns")
    parser.add_argument("--op", type=str, default="max", choices=["max", "min"],
                        help='reduce operation: "max" or "min" (default: max)')
    args = parser.parse_args()

    test_reduce_prop_nan(args.op, args.M, args.N)
