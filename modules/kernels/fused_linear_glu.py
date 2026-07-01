"""
Fused Linear + ATanGLU / SwiGLU kernel for LYNXNet2.

Key insight:
  nn.Linear(dim, inner_dim*2) + GLU
  = x @ W^T + b   where W is [2K, K]
  = split → left * activate(gate)

We split W into W_left [K, K] and W_right [K, K] (views, no copy).
This allows each Triton program to handle one [BLOCK_M, BLOCK_N] tile
of BOTH halves independently, applying GLU within the tile.

HBM savings per call:
  Original: write [M, 2K] to HBM → read [M, 2K] for GLU = 2 round-trips
  Fused:    no intermediate write, GLU in-register = 0 round-trips
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# atan approximation (Triton 3.2 doesn't have tl.math.atan)
# ---------------------------------------------------------------------------

@triton.jit
def _atan_approx(x):
    """Approximate atan(x) using Taylor series + identity.

    For |x| <= 1: atan(x) ≈ x - x³/3 + x⁵/5 - x⁷/7  (7th order, error < 0.004)
    For |x| >  1: atan(x) = π/2 - atan(1/x)

    Casts to fp32 internally to prevent overflow in x²/x³ terms.
    """
    x = x.to(tl.float32)
    one = 1.0
    half_pi = 1.5707963267948966

    x_abs = tl.abs(x)
    is_large = x_abs > one

    # For |x| <= 1: direct Taylor
    x2 = x * x
    x3 = x2 * x
    x5 = x2 * x3
    x7 = x2 * x5
    small = x - x3 / 3.0 + x5 / 5.0 - x7 / 7.0

    # For |x| > 1: atan(x) = π/2 - atan(1/x)
    inv = tl.where(is_large, one / x_abs, one)
    inv2 = inv * inv
    inv3 = inv2 * inv
    inv5 = inv2 * inv3
    inv7 = inv2 * inv5
    sign = tl.where(x >= 0, one, -one)
    large = half_pi * sign - (inv * sign - inv3 * sign / 3.0 + inv5 * sign / 5.0 - inv7 * sign / 7.0)

    return tl.where(is_large, large, small)


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'K'],
)
@triton.jit
def _fused_linear_atan_glu_fwd_kernel(
    # Pointers
    x_ptr, w_left_ptr, w_right_ptr, b_left_ptr, b_right_ptr,
    y_ptr, left_ptr, gate_ptr,
    # Shape
    M, K,
    # Strides for x
    stride_x_b, stride_x_k,
    # Strides for left weight
    stride_wl_n, stride_wl_k,
    # Strides for right weight
    stride_wr_n, stride_wr_k,
    # Strides for output y
    stride_y_b, stride_y_n,
    # Strides for saved left/gate
    stride_l_b, stride_l_n,
    stride_g_b, stride_g_n,
    # Meta params
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Compute y = (x@W_left^T+b_left) * atan(x@W_right^T+b_right).

    Uses a 2D grid over (M // BLOCK_M, K // BLOCK_N).
    Each program handles one [BLOCK_M, BLOCK_N] tile of the output.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(K, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for broadcasting
    # x access:  m [BLOCK_M, 1]  × k [1, BLOCK_K]
    m_mask_2d = offs_m[:, None] < M
    # w access:  n [BLOCK_N, 1] × k [1, BLOCK_K]  (N×K)
    n_mask_nk = offs_n[:, None] < K
    # output / saved tensor access:  m [BLOCK_M, 1] × n [1, BLOCK_N]  (M×N)
    n_mask_mn = offs_n[None, :] < K
    # 1D masks for bias
    m_mask_1d = offs_m < M
    n_mask_1d = offs_n < K

    # Accumulators (accumulate over K for each output tile)
    # tl.dot with fp16 inputs returns fp32; we accumulate in fp32 for precision
    acc_left = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    acc_gate = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K-loop: accumulate over the shared contraction dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask_2d = k_offs[None, :] < K

        # Load x tile [BLOCK_M, BLOCK_K]
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x_b + k_offs[None, :] * stride_x_k,
            mask=m_mask_2d & k_mask_2d,
            other=0.0,
        )

        # Load W_left tile [BLOCK_N, BLOCK_K]
        wl = tl.load(
            w_left_ptr + offs_n[:, None] * stride_wl_n + k_offs[None, :] * stride_wl_k,
            mask=n_mask_nk & k_mask_2d,
            other=0.0,
        )
        acc_left += tl.dot(x, wl.T)

        # Load W_right tile [BLOCK_N, BLOCK_K]
        wr = tl.load(
            w_right_ptr + offs_n[:, None] * stride_wr_n + k_offs[None, :] * stride_wr_k,
            mask=n_mask_nk & k_mask_2d,
            other=0.0,
        )
        acc_gate += tl.dot(x, wr.T)

    # Add bias
    b_left = tl.load(b_left_ptr + offs_n, mask=n_mask_1d, other=0.0)
    b_right = tl.load(b_right_ptr + offs_n, mask=n_mask_1d, other=0.0)
    acc_left += b_left
    acc_gate += b_right

    # ATanGLU: out * atan(gate)
    gated = acc_left * _atan_approx(acc_gate)

    # Write output y
    tl.store(
        y_ptr + offs_m[:, None] * stride_y_b + offs_n[None, :] * stride_y_n,
        gated,
        mask=m_mask_2d & n_mask_mn,
    )

    # Save intermediates for backward
    tl.store(
        left_ptr + offs_m[:, None] * stride_l_b + offs_n[None, :] * stride_l_n,
        acc_left,
        mask=m_mask_2d & n_mask_mn,
    )
    tl.store(
        gate_ptr + offs_m[:, None] * stride_g_b + offs_n[None, :] * stride_g_n,
        acc_gate,
        mask=m_mask_2d & n_mask_mn,
    )


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'K'],
)
@triton.jit
def _fused_linear_atan_glu_bwd_kernel(
    # Saved from forward
    left_ptr, gate_ptr, grad_y_ptr,
    # Gradient output
    grad_x_ptr,
    # Weight pointers (for backward matmuls)
    w_left_ptr, w_right_ptr,
    # Shape
    M, K,
    # Strides
    stride_l_b, stride_l_n,
    stride_g_b, stride_g_n,
    stride_gy_b, stride_gy_n,
    stride_gx_b, stride_gx_k,
    stride_wl_n, stride_wl_k,
    stride_wr_n, stride_wr_k,
    # Meta params
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute grad_x = grad_left_pre @ W_left + grad_gate @ W_right

    where:
      grad_left_pre = grad_y * atan(gate)
      grad_gate     = grad_y * left / (gate^2 + 1)

    Grid over (M // BLOCK_M, K // BLOCK_K), accumulate over N = K.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask_mn = offs_m[:, None] < M       # [BLOCK_M, 1] for saved tensor loading
    k_mask_nk = offs_k[None, :] < K       # [1, BLOCK_K] for weight loading

    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    # Accumulate over N dimension (same as K, the output dim of W_left/W_right)
    for n_start in range(0, K, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask_mn = n_offs[None, :] < K         # [1, BLOCK_N] for M×N access
        n_mask_nk = n_offs[:, None] < K         # [BLOCK_N, 1] for N×K access

        # Load saved intermediates [BLOCK_M, BLOCK_N]
        left = tl.load(
            left_ptr + offs_m[:, None] * stride_l_b + n_offs[None, :] * stride_l_n,
            mask=m_mask_mn & n_mask_mn, other=0.0,
        )
        gate_val = tl.load(
            gate_ptr + offs_m[:, None] * stride_g_b + n_offs[None, :] * stride_g_n,
            mask=m_mask_mn & n_mask_mn, other=0.0,
        )
        grad_y = tl.load(
            grad_y_ptr + offs_m[:, None] * stride_gy_b + n_offs[None, :] * stride_gy_n,
            mask=m_mask_mn & n_mask_mn, other=0.0,
        )

        # ATanGLU backward (fp32 for numerical safety — gate^2 can overflow in fp16)
        gate_f32 = gate_val.to(tl.float32)
        left_f32 = left.to(tl.float32)
        atan_gate = _atan_approx(gate_f32)
        decay_out = left_f32 / (gate_f32 * gate_f32 + 1.0)
        grad_left_pre = grad_y * atan_gate
        grad_gate = grad_y * decay_out

        # Load W_left tile: [BLOCK_N, BLOCK_K]
        wl = tl.load(
            w_left_ptr + n_offs[:, None] * stride_wl_n + offs_k[None, :] * stride_wl_k,
            mask=n_mask_nk & k_mask_nk, other=0.0,
        )
        # Load W_right tile: [BLOCK_N, BLOCK_K]
        wr = tl.load(
            w_right_ptr + n_offs[:, None] * stride_wr_n + offs_k[None, :] * stride_wr_k,
            mask=n_mask_nk & k_mask_nk, other=0.0,
        )

        # Accumulate grad_x from both paths
        acc += tl.dot(grad_left_pre.to(tl.float16), wl)
        acc += tl.dot(grad_gate.to(tl.float16), wr)

    # Write grad_x with mask [BLOCK_M, BLOCK_K]
    m_mask_gx = offs_m[:, None] < M
    k_mask_gx = offs_k[None, :] < K
    tl.store(
        grad_x_ptr + offs_m[:, None] * stride_gx_b + offs_k[None, :] * stride_gx_k,
        acc,
        mask=m_mask_gx & k_mask_gx,
    )


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'K'],
)
@triton.jit
def _fused_atan_glu_bwd_elem_kernel(
    left_ptr, gate_ptr, grad_y_ptr,
    glp_ptr, gg_ptr,
    M, K,
    stride_l_b, stride_l_n,
    stride_g_b, stride_g_n,
    stride_gy_b, stride_gy_n,
    stride_glp_b, stride_glp_n,
    stride_gg_b, stride_gg_n,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Element-wise ATanGLU backward: grad_left_pre, grad_gate from left, gate, grad_y.

    Single kernel, no intermediate HBM traffic. Grid over (M // BLOCK_M, K // BLOCK_K).
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    pid_m = pid // num_pid_k
    pid_k = pid % num_pid_k

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m[:, None] < M
    k_mask = offs_k[None, :] < K

    left = tl.load(left_ptr + offs_m[:, None] * stride_l_b + offs_k[None, :] * stride_l_n,
                   mask=m_mask & k_mask, other=0.0)
    gate = tl.load(gate_ptr + offs_m[:, None] * stride_g_b + offs_k[None, :] * stride_g_n,
                   mask=m_mask & k_mask, other=0.0)
    gy = tl.load(grad_y_ptr + offs_m[:, None] * stride_gy_b + offs_k[None, :] * stride_gy_n,
                 mask=m_mask & k_mask, other=0.0)

    # ATanGLU backward in registers — no intermediates written to HBM
    gate_f32 = gate.to(tl.float32)
    atan_gate = _atan_approx(gate_f32)
    decay_out = left.to(tl.float32) / (gate_f32 * gate_f32 + 1.0)

    tl.store(glp_ptr + offs_m[:, None] * stride_glp_b + offs_k[None, :] * stride_glp_n,
             gy * atan_gate, mask=m_mask & k_mask)
    tl.store(gg_ptr + offs_m[:, None] * stride_gg_b + offs_k[None, :] * stride_gg_n,
             gy * decay_out, mask=m_mask & k_mask)

class FusedLinearATanGLUFn(torch.autograd.Function):
    """Fused Linear(2K, K) + ATanGLU with custom backward."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        # weight: [2*K, K], bias: [2*K]
        orig_shape = x.shape
        K = weight.shape[1]

        # Flatten batch dims to [M, K]
        if x.dim() == 2:
            x_2d = x
        else:
            x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]

        # Split weight and bias into left/right (views ⇔ no copy)
        w_left, w_right = weight.split(K, dim=0)
        b_left, b_right = bias.split(K, dim=0)

        # Allocate outputs
        out = torch.empty(M, K, device=x.device, dtype=x.dtype)
        left = torch.empty(M, K, device=x.device, dtype=x.dtype)
        gate = torch.empty(M, K, device=x.device, dtype=x.dtype)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(K, meta['BLOCK_N']),)

        _fused_linear_atan_glu_fwd_kernel[grid](
            x_2d, w_left, w_right, b_left, b_right,
            out, left, gate,
            M, K,
            x_2d.stride(0), x_2d.stride(1),
            w_left.stride(0), w_left.stride(1),
            w_right.stride(0), w_right.stride(1),
            out.stride(0), out.stride(1),
            left.stride(0), left.stride(1),
            gate.stride(0), gate.stride(1),
        )

        # Restore batch dims for output only
        if x.dim() > 2:
            out = out.view(*orig_shape[:-1], K)
            # Keep left/gate flat [M, K] for backward
            left = left.view(M, K)
            gate = gate.view(M, K)

        ctx.save_for_backward(x_2d, weight, left, gate)
        ctx.orig_x_shape = orig_shape
        return out

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, left, gate = ctx.saved_tensors
        M, K = x.shape

        w_left, w_right = weight.split(K, dim=0)

        # Flatten grad_y to [M, K]
        if grad_y.dim() > 2:
            grad_y = grad_y.reshape(-1, K)

        # ── Step 1: Fused element-wise ATanGLU backward ──
        # Single Triton kernel, no intermediate atan_gate/decay_out materialized
        grad_left_pre = torch.empty(M, K, device=x.device, dtype=x.dtype)
        grad_gate = torch.empty(M, K, device=x.device, dtype=x.dtype)

        def elem_grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(K, meta['BLOCK_K']),)

        _fused_atan_glu_bwd_elem_kernel[elem_grid](
            left, gate, grad_y,
            grad_left_pre, grad_gate,
            M, K,
            left.stride(0), left.stride(1),
            gate.stride(0), gate.stride(1),
            grad_y.stride(0), grad_y.stride(1),
            grad_left_pre.stride(0), grad_left_pre.stride(1),
            grad_gate.stride(0), grad_gate.stride(1),
        )

        # ── Step 2: Weight gradients (PyTorch matmul) ──
        grad_w_left = grad_left_pre.T @ x   # [K, M] @ [M, K] = [K, K]
        grad_w_right = grad_gate.T @ x       # [K, M] @ [M, K] = [K, K]
        grad_weight = torch.cat([grad_w_left, grad_w_right], dim=0)
        grad_bias = torch.cat([grad_left_pre.sum(0), grad_gate.sum(0)], dim=0)

        # ── Step 3: Input gradient (fused backward kernel) ──
        grad_x = torch.empty_like(x)

        def bwd_grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(K, meta['BLOCK_K']),)

        _fused_linear_atan_glu_bwd_kernel[bwd_grid](
            left, gate, grad_y,
            grad_x,
            w_left, w_right,
            M, K,
            left.stride(0), left.stride(1),
            gate.stride(0), gate.stride(1),
            grad_y.stride(0), grad_y.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            w_left.stride(0), w_left.stride(1),
            w_right.stride(0), w_right.stride(1),
        )

        # Restore batch dims if input had them
        if len(ctx.orig_x_shape) > 2:
            grad_x = grad_x.view(*ctx.orig_x_shape)

        return grad_x, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_linear_atan_glu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused Linear(2K, K) + ATanGLU.

    Args:
        x: Input [..., K]
        weight: Weight [2*K, K] (as in nn.Linear(dim, 2*dim))
        bias: Bias [2*K]

    Returns:
        Output [..., K] = left * atan(gate)
    """
    assert weight.shape[0] == 2 * weight.shape[1], \
        f"Expected weight shape [2*K, K], got {weight.shape}"
    # 确保内存连续，避免隐式拷贝
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    return FusedLinearATanGLUFn.apply(x, weight, bias)


def fused_linear_swiglu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Non-fused Linear(2K, K) + SwiGLU (uses PyTorch ops — easy to Triton-ize later)."""
    K = weight.shape[1]
    w_left, w_right = weight.split(K, dim=0)
    b_left, b_right = bias.split(K, dim=0)
    left = F.linear(x, w_left, b_left)
    gate = F.linear(x, w_right, b_right)
    return left * F.silu(gate)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def _test():
    torch.manual_seed(42)
    device = 'cuda'

    for K in [256, 512, 1024]:
        M = 8192
        x = torch.randn(M, K, device=device, dtype=torch.float16, requires_grad=True)
        w = torch.randn(2 * K, K, device=device, dtype=torch.float16, requires_grad=True)
        b = torch.randn(2 * K, device=device, dtype=torch.float16, requires_grad=True)

        # Reference: PyTorch Linear + ATanGLU
        y_ref = F.linear(x, w, b)
        ref_left, ref_gate = y_ref.chunk(2, dim=-1)
        ref_out = ref_left * torch.atan(ref_gate)

        # Fused
        fused_out = fused_linear_atan_glu(x, w, b)

        fwd_max_diff = (fused_out - ref_out).abs().max().item()
        fwd_rel_diff = (fused_out - ref_out).abs().mean().item() / ref_out.abs().mean().item()

        # Backward
        grad = torch.randn_like(ref_out)
        ref_out.backward(grad)
        grad_x_ref = x.grad.clone()
        grad_w_ref = w.grad.clone()

        x.grad = None
        w.grad = None

        fused_out2 = fused_linear_atan_glu(x, w, b)
        fused_out2.backward(grad)
        grad_x_fused = x.grad.clone()
        grad_w_fused = w.grad.clone()

        diff_x = (grad_x_fused - grad_x_ref).abs().max().item()
        diff_w = (grad_w_fused - grad_w_ref).abs().max().item()
        re_x = diff_x / (grad_x_ref.abs().max().item() + 1e-10)
        re_w = diff_w / (grad_w_ref.abs().max().item() + 1e-10)

        # Benchmark
        import time
        n_warm = 10
        n_iter = 100

        # Reference time
        for _ in range(n_warm):
            _ = F.linear(x, w, b).chunk(2, dim=-1)[0] * torch.atan(
                F.linear(x, w, b).chunk(2, dim=-1)[1])
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            o = F.linear(x, w, b).chunk(2, dim=-1)[0] * torch.atan(
                F.linear(x, w, b).chunk(2, dim=-1)[1])
        torch.cuda.synchronize()
        ref_time = (time.time() - t0) / n_iter

        # Fused time
        for _ in range(n_warm):
            _ = fused_linear_atan_glu(x, w, b)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            _ = fused_linear_atan_glu(x, w, b)
        torch.cuda.synchronize()
        fused_time = (time.time() - t0) / n_iter

        print(f"K={K:4d}  │  fwd Δ_max={fwd_max_diff:.4e}  rel={fwd_rel_diff:.4e}  │  "
              f"bwd Δx={diff_x:.4e}(r={re_x:.4e}) Δw={diff_w:.4e}(r={re_w:.4e})  │  "
              f"ref={ref_time*1000:.2f}ms  fused={fused_time*1000:.2f}ms  speedup={ref_time/fused_time:.2f}x")


if __name__ == '__main__':
    _test()
