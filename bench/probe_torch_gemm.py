# GEMM 形状矩阵对拍（torch/cuBLAS 侧，CUDA event 计时，best-of-N）
# 与 layer_bench --op matmul/matmul_at/matmul_bt 同 (m,k,n) × 变体矩阵。
# 用法: python bench/probe_torch_gemm.py
import torch, time

DEV = torch.device("cuda")
DTYPE = torch.float32
ITERS, WARMUP = 50, 10

TRIPLES = [
    (192, 64, 16384),      # nn proj fwd 形状
    (16384, 64, 192),      # torch proj fwd 形状
    (64, 192, 16384),      # nn proj dgrad 形状
    (16384, 192, 64),      # torch proj dgrad 形状
    (192, 16384, 64),      # wgrad 形状（两侧同 mkn，转置变体不同）
    (8208, 64, 16384),     # nn head fwd
    (16384, 64, 8208),     # torch head fwd
    (64, 8208, 16384),     # nn head dgrad
    (16384, 8208, 64),     # torch head dgrad
    (8208, 16384, 64),     # head wgrad
    (4096, 4096, 4096),    # 方阵
    (256, 64, 16384),      # nn ffn1 fwd
    (16384, 64, 256),      # torch ffn1 fwd
]

def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(ITERS):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best  # ms

def gflops(m, k, n, ms):
    return 2.0 * m * k * n / (ms * 1e-3) / 1e9

rows = []
for (m, k, n) in TRIPLES:
    A = torch.randn(m, k, device=DEV, dtype=DTYPE)
    B = torch.randn(k, n, device=DEV, dtype=DTYPE)
    # plain: A(m,k) @ B(k,n)
    t_plain = bench(lambda: torch.mm(A, B))
    # transA: A 存 (k,m)，计算 A^T @ B
    At = torch.randn(k, m, device=DEV, dtype=DTYPE)
    t_ta = bench(lambda: torch.mm(At.t(), B))
    # transB: B 存 (n,k)，计算 A @ B^T
    Bt = torch.randn(n, k, device=DEV, dtype=DTYPE)
    t_tb = bench(lambda: torch.mm(A, Bt.t()))
    rows.append((m, k, n, t_plain, t_ta, t_tb))
    print(f"({m:5d},{k:5d},{n:5d})  plain {t_plain:8.4f} ms {gflops(m,k,n,t_plain):8.1f} G  "
          f"transA {t_ta:8.4f} ms {gflops(m,k,n,t_ta):8.1f} G  "
          f"transB {t_tb:8.4f} ms {gflops(m,k,n,t_tb):8.1f} G", flush=True)

print("\n== CSV ==")
print("m,k,n,plain_ms,transA_ms,transB_ms")
for (m, k, n, p, ta, tb) in rows:
    print(f"{m},{k},{n},{p:.4f},{ta:.4f},{tb:.4f}")
