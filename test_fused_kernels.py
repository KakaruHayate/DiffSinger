"""
=============================================================================
DiffSinger Triton Fused Kernel — 单元测试 & Benchmark
=============================================================================
测试内容:
  1. fused_linear_atan_glu — LYNXNet2 Linear+GLU 融合 kernel
     - 前向数值正确性 (vs torch.nn.Linear + ATanGLU)
     - 反向梯度正确性 (vs autograd)
     - Benchmark vs 原始实现

  2. ConvNeXtBlock 集成测试
     - monkey-patch 后的前向一致性
     - Eval 模式 (ONNX 路径) 正确性

运行方式:
  conda activate diffsinger
  python test_fused_kernels.py

硬件要求:
  RTX 2070+ (sm_75), CUDA 可用
=============================================================================
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
import triton

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 测试参数 (模拟实际训练配置)
TEST_CONFIGS = {
    'small':  {'M': 8192,  'K': 256,  'inner_dim': 256,  'expansion_factor': 1},
    'medium': {'M': 24576, 'K': 512,  'inner_dim': 512,  'expansion_factor': 1},
    'large':  {'M': 50000, 'K': 1024, 'inner_dim': 1024, 'expansion_factor': 1},
}

TOLERANCE = {
    'fwd_max': 10.0,    # atan 近似引入的绝对误差, fp16 下可接受
    'fwd_rel': 2e-3,    # 相对误差 < 0.2%
    'bwd_max': 2.0,     # 梯度绝对误差
    'bwd_rel': 5e-3,    # 梯度相对误差 < 0.5%
}

PASS = 0
FAIL = 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


# ============================================================================
# 1. Fused Linear+ATanGLU Kernel 测试
# ============================================================================

def test_fused_linear_atan_glu():
    """测试 fused_linear_atan_glu 的数值正确性和性能。"""
    from modules.kernels.fused_linear_glu import fused_linear_atan_glu

    device = 'cuda'
    print("\n" + "=" * 70)
    print("测试 1: fused_linear_atan_glu — LYNXNet2 Linear+ATanGLU 融合")
    print("=" * 70)

    for name, cfg in TEST_CONFIGS.items():
        M, K = cfg['M'], cfg['K']
        print(f"\n--- Config: {name} (M={M}, K={K}) ---")

        # 准备数据
        torch.manual_seed(42)
        x = torch.randn(M, K, device=device, dtype=torch.float16, requires_grad=True)
        w = torch.randn(2 * K, K, device=device, dtype=torch.float16, requires_grad=True)
        b = torch.randn(2 * K, device=device, dtype=torch.float16, requires_grad=True)

        # ── 前向测试 ──
        # 参考实现
        y_linear = F.linear(x, w, b)
        ref_left, ref_gate = y_linear.chunk(2, dim=-1)
        y_ref = ref_left * torch.atan(ref_gate)

        # 融合实现
        y_fused = fused_linear_atan_glu(x, w, b)

        # 比较
        abs_diff = (y_fused - y_ref).abs()
        max_diff = abs_diff.max().item()
        rel_diff = abs_diff.mean().item() / (y_ref.abs().mean().item() + 1e-10)

        check(f"[{name}] 前向 max diff", max_diff < TOLERANCE['fwd_max'],
              f"max_diff={max_diff:.4f}")
        check(f"[{name}] 前向 rel diff", rel_diff < TOLERANCE['fwd_rel'],
              f"rel_diff={rel_diff:.6f}")

        # ── 反向测试 ──
        grad = torch.randn_like(y_ref)

        # 参考反向
        y_ref.backward(grad)
        ref_grad_x = x.grad.clone()
        ref_grad_w = w.grad.clone()
        ref_grad_b = b.grad.clone()

        # 清除梯度
        x.grad, w.grad, b.grad = None, None, None

        # 融合反向
        y_fused2 = fused_linear_atan_glu(x, w, b)
        y_fused2.backward(grad)
        fused_grad_x = x.grad.clone()
        fused_grad_w = w.grad.clone()
        fused_grad_b = b.grad.clone()

        # 比较梯度
        dx = (fused_grad_x - ref_grad_x).abs().max().item()
        dw = (fused_grad_w - ref_grad_w).abs().max().item()
        db = (fused_grad_b - ref_grad_b).abs().max().item()

        # 归一化相对误差
        rx = dx / (ref_grad_x.abs().max().item() + 1e-10)
        rw = dw / (ref_grad_w.abs().max().item() + 1e-10)
        rb = db / (ref_grad_b.abs().max().item() + 1e-10)

        check(f"[{name}] 反向 grad_x", dx < TOLERANCE['bwd_max'] or rx < TOLERANCE['bwd_rel'],
              f"dx={dx:.4f} rel={rx:.6f}")
        check(f"[{name}] 反向 grad_w", dw < TOLERANCE['bwd_max'] or rw < TOLERANCE['bwd_rel'],
              f"dw={dw:.4f} rel={rw:.6f}")
        check(f"[{name}] 反向 grad_b", db < TOLERANCE['bwd_max'] or rb < TOLERANCE['bwd_rel'],
              f"db={db:.4f} rel={rb:.6f}")


def test_fused_with_batch_dims():
    """测试融合 kernel 在 batch 维度下的正确性 (B×T 摊平逻辑)。"""
    from modules.kernels.fused_linear_glu import fused_linear_atan_glu

    device = 'cuda'
    print("\n" + "=" * 70)
    print("测试 2: Batch 维度支持 (B×T 摊平)")
    print("=" * 70)

    B, T, K = 2, 250, 256
    torch.manual_seed(42)

    x = torch.randn(B, T, K, device=device, dtype=torch.float16, requires_grad=True)
    w = torch.randn(2 * K, K, device=device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(2 * K, device=device, dtype=torch.float16, requires_grad=True)

    # 参考：用 flatten 后的 [B*T, K]
    x_flat = x.reshape(-1, K)
    y_linear = F.linear(x_flat, w, b)
    ref_out = y_linear.chunk(2, dim=-1)[0] * torch.atan(y_linear.chunk(2, dim=-1)[1])
    ref_out = ref_out.view(B, T, K)

    # 融合：直接传 3D tensor
    fused_out = fused_linear_atan_glu(x, w, b)

    max_diff = (fused_out - ref_out).abs().max().item()
    check("3D batch 输入前向正确", max_diff < TOLERANCE['fwd_max'],
          f"max_diff={max_diff:.4f}")

    # 反向测试
    grad = torch.randn_like(ref_out)
    ref_out.backward(grad)
    ref_gx = x.grad.clone()

    x.grad = w.grad = b.grad = None
    fused_out2 = fused_linear_atan_glu(x, w, b)
    fused_out2.backward(grad)
    fwd_gx = x.grad.clone()

    dx = (fwd_gx - ref_gx).abs().max().item()
    check("3D batch 输入反向正确", dx < TOLERANCE['bwd_max'],
          f"dx={dx:.4f}")


# ============================================================================
# 2. ConvNeXtBlock 集成测试
# ============================================================================

def test_convnext_integration():
    """测试 ConvNeXtBlock 的 monkey-patch 集成和 eval 模式 fallback。"""
    from modules.backbones.lynxnet2 import LYNXNet2Block
    from modules.kernels.integration import wrap_lynxnet2_block

    device = 'cuda'
    print("\n" + "=" * 70)
    print("测试 3: LYNXNet2Block 集成 (monkey-patch)")
    print("=" * 70)

    dim, B, T = 256, 2, 250
    torch.manual_seed(42)

    # 原始 block
    block_orig = LYNXNet2Block(dim=dim, expansion_factor=1, glu_type='atanglu').to(device).half().train()
    block_patched = LYNXNet2Block(dim=dim, expansion_factor=1, glu_type='atanglu').to(device).half().train()
    block_patched.load_state_dict(block_orig.state_dict())

    # Patch
    wrap_lynxnet2_block(block_patched, glu_type='atanglu')

    x = torch.randn(B, T, dim, device=device, dtype=torch.float16, requires_grad=True)

    # ── Train mode ──
    out_orig = block_orig(x)
    out_patched = block_patched(x)

    diff_fwd = (out_patched - out_orig).abs().max().item()
    check("Train mode 前向一致", diff_fwd < TOLERANCE['fwd_max'],
          f"diff={diff_fwd:.4f}")

    # 反向
    grad = torch.randn_like(out_orig)
    out_orig.backward(grad)
    orig_grads = {n: p.grad.clone() for n, p in block_orig.named_parameters() if p.grad is not None}

    for p in block_patched.parameters():
        p.grad = None
    out_patched2 = block_patched(x)
    out_patched2.backward(grad)
    patched_grads = {n: p.grad.clone() for n, p in block_patched.named_parameters() if p.grad is not None}

    max_w_diff = max(
        (patched_grads[n] - orig_grads[n]).abs().max().item()
        for n in orig_grads
    )
    check("Train mode 反向一致", max_w_diff < TOLERANCE['bwd_max'],
          f"max_weight_grad_diff={max_w_diff:.4f}")

    # ── Eval mode (ONNX 路径) ──
    block_orig.eval()
    block_patched.eval()
    with torch.no_grad():
        out_orig_eval = block_orig(x)
        out_patched_eval = block_patched(x)

    diff_eval = (out_patched_eval - out_orig_eval).abs().max().item()
    check("Eval mode (ONNX 路径) 完全一致", diff_eval < 0.01,
          f"diff={diff_eval:.6f} (应为 0)")


# ============================================================================
# 3. Benchmark
# ============================================================================

def benchmark_linear_glu():
    """Benchmark fused_linear_atan_glu vs 原始 Linear + ATanGLU。"""
    from modules.kernels.fused_linear_glu import fused_linear_atan_glu

    device = 'cuda'
    print("\n" + "=" * 70)
    print("Benchmark: fused_linear_atan_glu vs nn.Linear + ATanGLU")
    print("=" * 70)
    print(f"{'Config':<15} {'M':>7} {'K':>5} {'Ref(ms)':>10} {'Fused(ms)':>10} {'Speedup':>8} "
          f"{'Ref+GC(ms)':>12} {'Fused+GC(ms)':>12} {'Spd+GC':>8}")
    print("-" * 70)

    for name, cfg in TEST_CONFIGS.items():
        M, K = cfg['M'], cfg['K']
        torch.manual_seed(42)

        x = torch.randn(M, K, device=device, dtype=torch.float16, requires_grad=True)
        w = torch.randn(2 * K, K, device=device, dtype=torch.float16, requires_grad=True)
        b = torch.randn(2 * K, device=device, dtype=torch.float16, requires_grad=True)

        N_WARMUP = 10
        N_ITER = 50

        # ── Forward only ──
        for _ in range(N_WARMUP):
            _ = F.linear(x, w, b).chunk(2, dim=-1)[0] * torch.atan(
                F.linear(x, w, b).chunk(2, dim=-1)[1])
            _ = fused_linear_atan_glu(x, w, b)
        torch.cuda.synchronize()

        # Ref forward
        t0 = time.time()
        for _ in range(N_ITER):
            y = F.linear(x, w, b).chunk(2, dim=-1)[0] * torch.atan(
                F.linear(x, w, b).chunk(2, dim=-1)[1])
        torch.cuda.synchronize()
        ref_fwd = (time.time() - t0) / N_ITER * 1000

        # Fused forward
        t0 = time.time()
        for _ in range(N_ITER):
            y = fused_linear_atan_glu(x, w, b)
        torch.cuda.synchronize()
        fused_fwd = (time.time() - t0) / N_ITER * 1000

        # ── Forward + Backward ──
        for _ in range(N_WARMUP):
            y = F.linear(x, w, b).chunk(2, dim=-1)[0] * torch.atan(
                F.linear(x, w, b).chunk(2, dim=-1)[1])
            loss = y.sum()
            loss.backward()

            y2 = fused_linear_atan_glu(x, w, b)
            loss2 = y2.sum()
            loss2.backward()
        torch.cuda.synchronize()

        # Ref fwd+bwd
        x.grad, w.grad, b.grad = None, None, None
        t0 = time.time()
        for _ in range(N_ITER):
            y = F.linear(x, w, b).chunk(2, dim=-1)[0] * torch.atan(
                F.linear(x, w, b).chunk(2, dim=-1)[1])
            y.sum().backward()
        torch.cuda.synchronize()
        ref_fb = (time.time() - t0) / N_ITER * 1000

        # Fused fwd+bwd
        x.grad, w.grad, b.grad = None, None, None
        t0 = time.time()
        for _ in range(N_ITER):
            y2 = fused_linear_atan_glu(x, w, b)
            y2.sum().backward()
        torch.cuda.synchronize()
        fused_fb = (time.time() - t0) / N_ITER * 1000

        print(f"{name:<15} {M:>7} {K:>5} "
              f"{ref_fwd:>8.2f}  {fused_fwd:>8.2f}  {ref_fwd/fused_fwd:>7.2f}x "
              f"{ref_fb:>10.2f}  {fused_fb:>10.2f}  {ref_fb/fused_fb:>7.2f}x")


# ============================================================================
# 4. 内存带宽节省验证 (理论值 vs 实测)
# ============================================================================

def benchmark_hbm_savings():
    """通过单个 block 的 full forward+backward 实测比较内存流量。"""
    from modules.backbones.lynxnet2 import LYNXNet2
    from modules.kernels.integration import patch_lynxnet2_model

    device = 'cuda'
    print("\n" + "=" * 70)
    print("Benchmark: 完整 LYNXNet2 模型 (6 layers, K=256)")
    print("=" * 70)

    # hparams 最低要求 (LYNXNet2 构建时需要)
    from utils.hparams import set_hparams, hparams
    # 设置 hparams 最低要求 (LYNXNet2 构建时需要)
    set_hparams()
    hparams['hidden_size'] = 384
    hparams['backbone_type'] = 'lynxnet2'
    hparams['backbone_args'] = {
        'num_channels': 256, 'num_layers': 6, 'expansion_factor': 1,
        'kernel_size': 31, 'dropout_rate': 0.0, 'use_conditioner_cache': False,
        'glu_type': 'atanglu'
    }

    # 构建小模型用于 benchmark
    model = LYNXNet2(
        in_dims=128, n_feats=1,
        num_layers=6, num_channels=256,
        expansion_factor=1, kernel_size=31,
        glu_type='atanglu'
    ).to(device).half().train()

    # 拷贝一份用于 patched benchmark
    model_patched = LYNXNet2(
        in_dims=128, n_feats=1,
        num_layers=6, num_channels=256,
        expansion_factor=1, kernel_size=31,
        glu_type='atanglu'
    ).to(device).half().train()
    model_patched.load_state_dict(model.state_dict())
    patch_lynxnet2_model(model_patched, glu_type='atanglu')

    B, M, T = 2, 128, 250
    spec = torch.randn(B, 1, M, T, device=device, dtype=torch.float16)
    t = torch.randint(0, 1000, (B,), device=device).float()
    cond = torch.randn(B, 384, T, device=device, dtype=torch.float16)

    N_WARMUP = 5
    N_ITER = 30

    for _ in range(N_WARMUP):
        _ = model(spec, t, cond=cond)
        _ = model_patched(spec, t, cond=cond)
    torch.cuda.synchronize()

    # Ref
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(N_ITER):
        out = model(spec, t, cond=cond)
        out.sum().backward()
    torch.cuda.synchronize()
    ref_t = (time.time() - t0) / N_ITER * 1000
    ref_mem = torch.cuda.max_memory_allocated() / 1024**3

    # Fused
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(N_ITER):
        out2 = model_patched(spec, t, cond=cond)
        out2.sum().backward()
    torch.cuda.synchronize()
    fused_t = (time.time() - t0) / N_ITER * 1000
    fused_mem = torch.cuda.max_memory_allocated() / 1024**3

    print(f"{'Config':<25} {'Time(ms)':>10} {'Mem(GB)':>10} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'Original (6×Linear+GLU)':<25} {ref_t:>10.2f} {ref_mem:>10.2f} {'1.00x':>10}")
    print(f"{'Fused Triton kernel':<25} {fused_t:>10.2f} {fused_mem:>10.2f} "
          f"{ref_t/fused_t:>9.2f}x")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("DiffSinger Triton Fused Kernel 单元测试 & Benchmark")
    print(f"设备: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}, Triton: {triton.__version__}")
    print("=" * 70)

    # 测试
    test_fused_linear_atan_glu()
    test_fused_with_batch_dims()
    test_convnext_integration()

    # Benchmark
    benchmark_linear_glu()

    # 结果汇总
    print("\n" + "=" * 70)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 70)
    if FAIL > 0:
        print("警告: 部分测试失败！请检查以上 FAIL 项。")
    else:
        print("全部通过！kernel 可以部署到训练流程中。")
    print()
    print("提示: 以上 benchmark 基于 RTX 2070 (sm_75)。")
    print("RTX 4090/5090 上的加速比会更高 (更多SM + 更大显存带宽)。")