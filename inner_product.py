import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 64}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE": 256}, num_stages=4, num_warps=4),
    ],
    key=["M", "N"],
)
@triton.jit
def inner_product_(
    m_ptr,
    v_ptr,
    out_ptr,
    M,
    N,
    stride_m0,
    stride_m1,
    stride_v0,
    stride_out0,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_m = pid  # each program id handles one row

    tl.assume(pid_m >= 0)
    tl.assume(pid_m < M)

    offs_n = tl.arange(0, BLOCK_SIZE)  # shape (BLOCK_SIZE,)
    offs_m = pid_m  # scalar

    # m_ptrs shape: (1, BLOCK_SIZE) after broadcasting
    m_ptrs = m_ptr + (offs_m * stride_m0 + offs_n[:, None] * stride_m1)
    # v_ptrs shape: (BLOCK_SIZE, 1)
    v_ptrs = v_ptr + (offs_n[:, None] * stride_v0)

    accum = tl.zeros((), dtype=tl.float32)

    # number of blocks along N dimension (works because N is scalar runtime int)
    for i in range(0, tl.cdiv(N, BLOCK_SIZE)):
        m = tl.load(
            m_ptrs, mask=offs_n[:, None] < N - i * BLOCK_SIZE, other=0.0
        )  # (1, BLOCK_SIZE)
        v = tl.load(
            v_ptrs, mask=offs_n[:, None] < N - i * BLOCK_SIZE, other=0.0
        )  # (BLOCK_SIZE, 1)

        # dot: both get broadcast to same shape; result is scalar accumulator
        accum += tl.sum(m * v)

        m_ptrs += BLOCK_SIZE * stride_m1
        v_ptrs += BLOCK_SIZE * stride_v0

    tl.store(out_ptr + (pid_m * stride_out0), accum.to(tl.float16))


def inner_product(m, v):
    assert m.shape[1] == v.shape[0], "Incompatible dimensions"
    assert v.is_contiguous(), "Vector v must be contiguous"
    M, N = m.shape
    N = v.shape[0]

    out = torch.empty((M,), device=m.device, dtype=torch.float16)

    def grid(META):
        return (M,)

    inner_product_[grid](
        m,
        v,
        out,
        M,
        N,
        m.stride(0),
        m.stride(1),
        v.stride(0),
        out.stride(0),
    )
    return out


def test_inner_product_fp16():
    torch.manual_seed(0)
    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((512,), device=DEVICE, dtype=torch.float16) - 0.5
    triton_output = inner_product(a, b)
    torch_output = torch.inner(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["M", "N"],  # Argument names to use as an x-axis for the plot
            x_vals=[
                128 * i for i in range(2, 36)
            ],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=["triton", "torch"],  # Label name for the lines
            line_names=["Triton", "Pytorch"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="inner-performance-"
            + (
                "fp16"
            ),  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    ]
)
def benchmark(M, N, provider):
    a = torch.randn((M, N), device=DEVICE, dtype=torch.float16)
    b = torch.randn((N,), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.inner(a, b), quantiles=quantiles
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: inner_product(a, b), quantiles=quantiles
        )

    def perf(ms):
        return 2 * M * N * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


test_inner_product_fp16()
benchmark.run(save_path="./", print_data=True)
