"""
Fused pwconv1 + GELU for ConvNeXtBlock.

Saves 1 write + 1 read of the [M, 4*C] intermediate (vs 2 writes + 2 reads).
Simpler than full pwconv1+GELU+pwconv2 fusion — avoids Triton slicing issues.

pwconv2 stays as a standard F.linear call (no fusion needed).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gelu_approx(x):
    return x * tl.sigmoid(1.702 * x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_linear_gelu_fwd_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,  # N = 4*C, K = C
    stride_x_b, stride_x_k,
    stride_w_n, stride_w_k,
    stride_out_b, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """y = GELU(x @ W^T + b)  — standard 2D-grid matmul + GELU fusion."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    n_mask_w = offs_n[:, None] < N    # [BLOCK_N, 1] for weight access
    n_mask_out = offs_n[None, :] < N  # [1, BLOCK_N] for output access

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs[None, :] < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x_b + k_offs[None, :] * stride_x_k,
            mask=m_mask & k_mask, other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_w_n + k_offs[None, :] * stride_w_k,
            mask=n_mask_w & k_mask, other=0.0,
        )
        acc += tl.dot(x, w.T)

    # Bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b

    # GELU
    acc = _gelu_approx(acc)

    tl.store(
        out_ptr + offs_m[:, None] * stride_out_b + offs_n[None, :] * stride_out_n,
        acc,
        mask=m_mask & n_mask_out,
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

class FusedLinearGELUFn(torch.autograd.Function):
    """Fused Linear(C, 4*C) + GELU."""

    @staticmethod
    def forward(ctx, x, w, b):
        orig_shape = x.shape
        K = w.shape[1]  # C
        N = w.shape[0]  # 4*C

        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]

        out = torch.empty(M, N, device=x.device, dtype=x.dtype)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        _fused_linear_gelu_fwd_kernel[grid](
            x_2d, w, b, out,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1),
        )

        if x.dim() > 2:
            out = out.view(*orig_shape[:-1], N)

        ctx.save_for_backward(x_2d, w)
        ctx.orig_x_shape = orig_shape
        return out

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        if grad_y.dim() > 2:
            grad_y = grad_y.reshape(-1, grad_y.shape[-1])

        # Recompute pre-GELU activation
        h = F.linear(x, w)  # [M, 4*C]

        # GELU backward via sigmoid approximation (matching forward)
        s = torch.sigmoid(1.702 * h)
        grad_h = grad_y * (s + h * 1.702 * s * (1 - s))

        # Weight gradients
        grad_w = grad_h.T @ x
        grad_b = grad_h.sum(0)
        grad_x = grad_h @ w

        if ctx.orig_x_shape and len(ctx.orig_x_shape) > 2:
            grad_x = grad_x.view(*ctx.orig_x_shape)

        return grad_x, grad_w, grad_b


def fused_linear_gelu(x, w, b):
    """Fused Linear(C, 4*C) + GELU.

    Args:
        x: [..., C]
        w: [4*C, C], b: [4*C]

    Returns:
        y = GELU(x @ W^T + b)  (shape [..., 4*C])
    """
    assert w.shape[0] == 4 * w.shape[1], f"W shape {w.shape} should be [4*C, C]"
    return FusedLinearGELUFn.apply(x.contiguous(), w.contiguous(), b.contiguous())


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _test():
    torch.manual_seed(42)
    device = 'cuda'

    for C in [128, 256, 512]:
        M = 8192 if C == 128 else (4096 if C == 256 else 2048)
        x = torch.randn(M, C, device=device, dtype=torch.float16, requires_grad=True)
        w = torch.randn(4 * C, C, device=device, dtype=torch.float16, requires_grad=True)
        b = torch.randn(4 * C, device=device, dtype=torch.float16, requires_grad=True)

        # Reference
        y_ref = F.gelu(F.linear(x, w, b))

        # Fused
        y = fused_linear_gelu(x, w, b)

        fwd_diff = (y - y_ref).abs().max().item()
        fwd_rel = fwd_diff / y_ref.abs().mean().item()

        # Backward
        grad = torch.randn_like(y_ref)
        y_ref.backward(grad)
        gx_ref = x.grad.clone()
        gw_ref = w.grad.clone()

        x.grad = w.grad = None
        y2 = fused_linear_gelu(x, w, b)
        y2.backward(grad)
        gx = x.grad.clone()
        gw = w.grad.clone()

        dx = (gx - gx_ref).abs().max().item()
        dw = (gw - gw_ref).abs().max().item()

        import time
        n_iter = 100
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            _ = F.gelu(F.linear(x, w, b))
        torch.cuda.synchronize()
        ref_t = (time.time() - t0) / n_iter

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            _ = fused_linear_gelu(x, w, b)
        torch.cuda.synchronize()
        fused_t = (time.time() - t0) / n_iter

        print(f"C={C:3d}  fwd_rel={fwd_rel:.4e}  "
              f"dx={dx:.4e} dw={dw:.4e}  "
              f"ref={ref_t*1000:.2f}ms fused={fused_t*1000:.2f}ms spd={ref_t/fused_t:.2f}x")


if __name__ == '__main__':
    _test()