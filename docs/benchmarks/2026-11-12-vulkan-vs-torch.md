# neuralnet.cpp (Vulkan) vs PyTorch (CUDA) 端到端训练基准（2026-11-12）

> 日期：2026-11-12 · 硬件：NVIDIA CMP 40HX（8 GB，本机唯一计算卡）· **全部数据本会话实测**
> **数据版本**：nn 与 torch 两侧**同日、同机、同卡、同语料**采集，无跨会话漂移。
> 与上一份报告 [`2026-10-06-vulkan-vs-torch.md`](2026-10-06-vulkan-vs-torch.md) 的关系见 §5：语料与词表均重建（字节与上轮不同，本轮两侧同用一份）；显存采样因 nvidia-smi/NVML 故障改用 PDH 计数器（§6.6）。
> 本轮 nn 侧为 HEAD `90dce64` 干净树 Release（运行期 JIT + `.nnk` 缓存，AOT 融合链已于 11-08 删除；引擎 `nn-cpu.dll` / `nn-gpu.dll` 同日重编）。

## 0. 结论速览

| 维度 | neuralnet.cpp (Vulkan) | PyTorch (CUDA) | 结论 |
|---|---|---|---|
| **训练速度**（f32 ↔ fp32，**主对比**） | **109.2 ms/step ≈ 150.1K tok/s**（复跑 112.2 / 146.0K） | **53.5 ms/step ≈ 304.5K tok/s**（复跑 53.8 / 303.5K） | **torch 快 2.03×**（复跑 2.08×） |
| 训练速度（双方各自可用的低精度） | 138.8 ms/step ≈ 118.1K tok/s（`--f16`，健康） | 33.5 ms/step ≈ 486.8K tok/s（fp16，稳定） | torch 快 4.12× |
| 训练速度（双方全 f16 ⚠） | **epoch 1 首步即 OOM 崩溃**（`VK_ERROR_OUT_OF_DEVICE_MEMORY`） | 33.5 ms/step ≈ 486.8K tok/s | **仅 torch 有效**；nn 训练无效 |
| **显存占用**（PDH 整卡峰值，同口径） | **f32 3237 / `--f16` 6972 / 全 f16 7365（崩溃前） MiB** | fp32 3741 / fp16 1915 MiB | f32：**nn 更省（0.87×）**；低精度：**nn 多 3.6× ⚠** |
| 训练稳定性（lr=1e-3，3 epoch） | **f32 与 `--f16` 均健康收敛**；全 f16 首步 OOM | fp32 / fp16 全程稳定 | f32 打平；nn 全 f16 不可用 |
| GPU 利用率（同配置 profile） | **本轮缺失**（设备侧 timestamp 表未回填，待查 §4.3） | — | 上轮 89.8% 无法复核 |

**两个关键发现**：

1. **f32 主差距稳定在 ≈2.0–2.1×，上轮结论成立**（§2、§5）。nn f32 相对 10-06 仅 +2.0%、torch fp32 +6.6%，比值 2.14× → 2.03×（复跑 2.08×），两侧 loss 曲线仍几乎重合，落在跨轮漂移内。torch fp16"病态慢"仍不复现（§3，probe forward 9.2 ms 而非 09-25 的 47.6 ms）。
2. **nn 的 f16 路径出现显存/性能回归 ⚠（本轮新发现，待查）**（§4.3、§5）。`--f16` 显存 2113（上轮 nvidia-smi）→ **6972 MiB**（同轮 PDH 与 f32 的口径差仅 ~3%，解释不了 3.3×）；e2e 慢 17%（118.4→138.8 ms/step）；profile 下同调用数（3687 次）的 `eval_expr` 单次均值 229.9→509.8 µs（**2.2×**）。全 f16 从上轮"跑完 3 epoch 但 -nan"恶化为**首步 OOM**。f32 路径不受影响。

---

## 1. 环境与配置

| 项 | neuralnet.cpp | PyTorch |
|---|---|---|
| 版本 | 本工作区 Release（clang、`-O3 -march=native`），HEAD `90dce64` 干净树；引擎 DLL 同日重编；运行期 JIT + `.nnk` 缓存 | 2.15.0+cu132（本轮新建 `.venv/`，Python 3.15.2） |
| 后端 | Vulkan GpuEngine（`--gpu=40HX`） | CUDA (cuBLAS/cuDNN) |
| 精度（**主对比**） | **f32（默认）** | **fp32（默认）** |
| 对照精度 | `--f16` = {param:f16, compute:f16, stable:f32, optimizer:f32}（健康）；全 f16 = 四字段全 f16（⚠ 本轮首步 OOM） | fp16 = `model.half()`，`adam eps=1e-4` |
| 优化器 | Adam，lr=1e-3，β=(0.9,0.999)，eps=1e-8 | 同超参 |

**模型（两侧同构）**：GPT，d_model=64，heads=4，layers=4，d_ff=256，seq_len=256，batch=64。
参数量：nn **1,275,280**（vocab 8208，与上轮同配置同词表）/ torch **1,273,216**（vocab 8192），差 0.16%。

**数据（本轮重建，与 10-06 字节不同）**：10-06 语料的来源文件已不在磁盘（`datasets/` 未入库）。本轮改从 HuggingFace `roneneldan/TinyStories` train 分割按行序取**前 4000 条**（datasets-server rows API），按 `scripts/download_tinystories.py` 同款清洗（换行/制表符 → 空白压缩）→ `datasets/tinystories_bench40.txt`，**4000 行 / 3,377,345 字节 / sha256 `0a2edd42958cfc7b…`**。与 10-06 记录（3,381,238 字节 / `584422376434996e…`）字节不同（体积差 0.12%）；**本会话 nn 与 torch 两侧同用这一份**。

- nn：BBPE 训练词表 8208 → **789,850 tokens**（上轮 789,828）→ 滑窗 3086 窗口 → **49 步/epoch**
- torch：HF BPE 8192 于本轮现场训练（上轮沿用的 `bench/raw/torch_bpe_8192.json` 已不在）→ **782,897 tokens**（上轮 785,820）→ 3059 窗口 → **48 步/epoch**

两侧分词器独立、token 流不同，**每步形状完全相同（64×256 = 16,384 位置）**，速度按 tok/s 归一。

**数据管线（与 10-06 同口径）**：每行独立 tokenize 后拼接；窗口 `pos = 0,256,…`，`x[t]=flow[pos+t]`，`y[t]=flow[pos+t+1]` 且仅 `t+1 < win_len` 参与 loss；每 epoch 打乱窗口；epoch 边界同步 GPU（nn drain loss 回读 / torch `synchronize`），epoch 内 host 构 batch 与 H2D 和 GPU 重叠。

**设备异常记录（不影响两侧同卡结论）**：Vulkan 与 CUDA 均枚举到 **NVIDIA CMP 40HX**（`nn_warmup` 与 `torch.cuda.get_device_name` 互证）；但 WMI 显示名报 "RTX 2060 SUPER"、`nvidia-smi` NVML 初始化失败（`Failed to initialize NVML: Unknown Error`）。见 §6.6/§6.7。

---

## 2. 端到端训练（主表）

**口径**：3 epoch，**稳态 = epoch 2–3 平均**（epoch 1 含初始化与时钟爬坡）；显存由 `bench/run_with_vram.ps1` 以 PDH `\GPU Adapter Memory(*)\Dedicated Usage` 采样整卡峰值（NVML 故障回落，§6.6）；每步 16,384 个位置。

| 配置 | 稳态 s/epoch | ms/step | tok/s | 显存峰 (PDH) | loss 曲线（e1→e2→e3） |
|---|---:|---:|---:|---:|---|
| **nn Vulkan f32**（**主对比**） | 5.35 | **109.2** | **150,059** | **3237 MiB** | 7.359 → 5.844 → 5.146 健康 |
| nn Vulkan f32（复跑） | 5.50 | 112.2 | 145,966 | 3453 MiB | 7.344 → 5.844 → 5.177 健康 |
| nn Vulkan `--f16`（推荐低精度 profile） | 6.80 | 138.8 | 118,061 | **6972 MiB ⚠** | 7.351 → 5.903 → 5.244 健康 |
| nn Vulkan 全 f16（⚠） | — | — | — | 7365 MiB（崩溃前 2 点） | **首步 OOM 崩溃（训练无效）** |
| **torch CUDA fp32**（**主对比**） | 2.57 | **53.5** | **304,534** | **3741 MiB** | 7.338 → 5.897 → 5.231 最稳 |
| torch CUDA fp32（复跑） | 2.58 | 53.8 | 303,464 | 3734 MiB | 7.392 → 5.925 → 5.257 |
| torch CUDA fp16 | 1.61 | 33.5 | 486,759 | 1915 MiB | 7.573 → 6.734 → 6.282 稳定 |

**读数**：

- **主对比（f32 ↔ fp32）**：torch 吞吐 **2.03×** 于 nn（304,534 / 150,059 = 2.03；复跑比 2.08×）；**显存 nn 更省**（3237 vs 3741 = 0.87×）。nn 两次复跑离散 ±3%，torch ±0.4%。
- **可用低精度对比**：nn `--f16` 118,061 vs torch fp16 486,759 → torch **4.12×**（上轮 3.89×）；显存 nn 6972 vs torch 1915 → **nn 多 3.6× ⚠**（上轮 2113 vs 1981，仅多 7%）——差距主要来自 nn 侧膨胀，见 §4.3。
- **双方全 f16**：nn 本轮首步 `vkAllocateMemory failed` OOM（`--log-interval 1` 下无任何 step 行完成）；torch fp16 正常。该行**仅 torch 与 nn 的崩溃前显存采样有效**。
- 两侧 f32 loss 曲线几乎重合（nn 7.359→5.146 / torch 7.338→5.231；复跑亦然），可交叉验证负载等价。
- 本表 7 行均来自本轮实测，**不含跨会话数据**。

---

## 3. torch 单步分解与 fp16 复核

`bench/probe_torch_step.py`（50 轮 best，CUDA event，fp16/half 路径，与 09-25 / 10-06 同一脚本）：

| 段 | 09-25 实测 | 10-06 实测 | **2026-11-12 实测** | 本轮 vs 上轮 |
|---|---:|---:|---:|---:|
| host 构 batch（CPU wall） | 0.12 ms | 0.125 ms | **0.111 ms** | 持平 |
| H2D（含构 batch） | 0.59 ms | 0.507 ms | **0.534 ms** | 持平 |
| forward（无 loss） | 47.64 ms | 8.554 ms | **9.225 ms** | +7.8% |
| forward + CE loss | 49.64 ms | 11.490 ms | **12.872 ms** | +12.0% |
| forward + loss + backward | 150.02 ms | 26.604 ms | **29.407 ms** | +10.5% |
| Adam step | 1.77 ms | 1.678 ms | **1.722 ms** | 持平 |
| zero_grad | 0.12 ms | 0.112 ms | **0.004 ms** | 持平 |
| full step（host wall / 双侧 sync） | 32.3 / 150.96 ms | 29.017 / 28.797 ms | **30.824 / 31.001 ms** | +7% |

**结论：09-25 记录的"torch fp16 病态（比 fp32 慢 3.3×）"本轮仍不复现（连续第二轮）。**

- 端到端口径：fp16 33.5 ms/step vs fp32 53.5 ms/step → **fp16 快 1.60×**（10-06 为 1.65×；09-25 旧记录为慢 3.16×）。
- 单步口径与 10-06 同量级（`fwd+loss+bwd` 26.604 → 29.407 ms，+10.5%），各未变段（host 构 batch、H2D、Adam、zero_grad）依旧持平——脚本与管线未变，差异仍在 GPU 执行侧的整体节奏（与端到端 +6.6% 同向）。
- torch 版本 2.14.0 → 2.15.0+cu132，驱动亦有更新（WMI 32.0.16.2101，§1），本轮不与 10-06 做逐毫秒对齐，只确认**量级与趋势**。

---

## 4. neuralnet 训练 step 剖析（`NN_PROFILE=1`）

与 §2 完全同配置（d64 / 4 层 / heads4 / d_ff256 / seq256 / batch64，同一 `.nndataset`，`--gpu=40HX`），f32 20 步 + 全 epoch 双跑、`--f16` 20 步对照。

产物：`build/prof_bench_20261112.txt`（f32 20 步）、`build/prof_bench_f16_20261112.txt`、`build/trace_bench_20261112.json`。

> ⚠ **与 10-06 §4 的表不可逐行对比**：profiler 的阶段命名与 host 侧归属自上轮后有演进（新增 `ce.fold`/`ce.grad` 等阶段、`copy_from` 的阻塞归属迁移），且**本轮设备侧 timestamp 表未回填（GPU busy% 缺失，待查 §4.2）**。本节只作同轮 f32↔f16 结构对照。

### 4.1 阶段构成（inclusive 墙钟，f32 20 步）

墙钟合计 **2326.13 ms / 20 步 = 116.3 ms/step**（e2e 实测 109.2 ms/step，profile 开销 +6.5%；整 epoch 口径 5554.83 ms / 49 步 = 113.4 ms/step，+3.8%）。

| stage | ms | %wall |
|---|---:|---:|
| forward | 2319.31 | 99.7% |
| └ fwd.layer#0 | 1208.06 | 51.9% |
| └ ce.fold | 519.49 | 22.3% |
| └ fwd.blk#2 | 325.33 | 14.0% |
| └ fwd.blk#1 | 306.27 | 13.2% |
| └ fwd.blk#0 | 303.77 | 13.1% |
| └ fwd.blk#3 | 39.98 | 1.7% |
| bwd.layer#0 | 110.69 | 4.8% |
| ce.grad | 65.47 | 2.8% |
| data | 6.14 | 0.3% |

（子行嵌套于 forward 内、彼此含重叠归属，不作求和。上轮占 85.4% 的 forward 与本轮 99.7% 因阶段口径不同不可直接比。）

### 4.2 原语净归属（exclusive）与设备侧

| op | ms | calls | avg µs | %wall |
|---|---:|---:|---:|---:|
| `eval_expr` | 847.72 | 3687 | 229.9 | 36.4% |
| `eval_expr_into` | 107.35 | 5824 | 18.4 | 4.6% |
| `zero` | 54.08 | 3065 | 17.6 | 2.3% |
| `eval_expr_reduce` | 29.95 | 1698 | 17.6 | 1.3% |
| `copy_from` | 24.53 | 121 | 202.7 | 1.1% |
| `create_tensor` | 18.11 | 422 | 42.9 | 0.8% |
| `rearrange_3d` | 13.74 | 984 | 14.0 | 0.6% |

- **设备侧（timestamp query）本轮未输出**：`core_profile.hpp` 表 4/表 5 需 `device` 桶非空，本轮 f32 与 f16、20 步与整 epoch 四次运行均无 GPU kernel 表 / per-key 表 / busy%（host 表与 trace JSON 正常）。`ts_enabled_ = prof::enabled() && queue_timestamp_valid_bits() > 0` 两条前提中 host 侧 `NN_PROFILE` 已生效，故嫌疑集中在队列时间戳能力读数或回填路径——**标注待查**，上轮 89.8% busy 的结论本轮无法复核。
- `copy_from` 从上轮 78.3% 降到 1.1%：上轮该值主要是**设备饱和下的 host 阻塞等待被记到调用点**；本轮阻塞归属迁移到 `eval_expr` 等执行路径（无设备表无法进一步归因）。两轮均不可把 host exclusive 表读作"传输时间"。
- 每步 dispatch 量级仍以 host 调用计：`eval_expr` 3687/20 ≈ 184 + `eval_expr_into` 5824/20 ≈ 291 ≈ **475 次/步**（上轮含 `fused_mm` 的设备口径 602 次/步，口径不同）。

### 4.3 f16 路径异常剖析（本轮新发现，待查）

**现象四件套（全部本轮实测）**：

1. **显存**：`--f16` e2e 峰 6972 MiB vs f32 3237 MiB（同 PDH 口径，**2.15×**）；上轮 nvidia-smi 口径下 `--f16` 仅 2113 MiB（比 f32 还省）。分适配器探针（`build/mem_probe.ps1`，20 步、200 ms、逐适配器）复现：f32 序列 394→**3237**→2724（步间回落），f16 序列 394→**6945**→6638（epoch 内持续高位，epoch 边界回落到 ~1394 后再爬升）。
2. **速度**：e2e `--f16` 138.8 ms/step，比 f32 **慢 27%**（上轮慢 10.5%；10-14 曾实测 P2-11/P2-13 后 f16 反快 1.4%，见 `docs/internal/18` C9——该收益在当前 HEAD 的 40HX 上已不存在）。
3. **kernel 级**：f16 profile 20 步 `eval_expr` **3687 次、avg 509.8 µs** vs f32 同 **3687 次、avg 229.9 µs**——**调用数逐一同、单次 2.22× 慢**；f16 墙钟 4570.03 vs f32 2326.13 ms（注：f16 剖面含池增长期，profile 开销大于 f32，绝对值只作结构对照）。
4. **全 f16**：由上轮"3 epoch 跑完、epoch1 起 -nan"恶化为**首个 update step 内 OOM**（`vkAllocateMemory failed: VK_ERROR_OUT_OF_DEVICE_MEMORY`；`--log-interval 1` 无任何 step 行，2 个采样点峰 7365 MiB）。

**未做根因二分**（本轮范围是基准复测；嫌疑方向：f16 边界 cast 的临时张量与池底材增长、in-kernel f16 变体的生成/执行路径、驱动更新后的行为差异）。f32 路径与 torch 侧均不受影响。

---

## 5. 与 2026-10-06 报告的差异

| 指标 | 10-06 | **2026-11-12** | 判定 |
|---|---:|---:|---|
| nn f32 ms/step | 107.1 | **109.2**（复跑 112.2） | +2.0%，漂移内 |
| torch fp32 ms/step | 50.2 | **53.5**（复跑 53.8） | +6.6%，漂移内 |
| **f32 吞吐比值** | **2.14×** | **2.03×**（复跑 2.08×） | **结论稳定（≈2.0–2.1×）** |
| nn f32 loss 收敛 | 7.335 → 5.868 → 5.171 | 7.359 → 5.844 → 5.146 | **复现**（语料重建后仍重合） |
| torch fp32 loss 收敛 | 7.383 → 5.897 → 5.342 | 7.338 → 5.897 → 5.231 | **复现** |
| torch fp16 ms/step | 30.4（比 fp32 快 1.65×） | 33.5（快 1.60×） | 趋势复现 |
| torch fp16 forward（probe） | 8.554 ms | 9.225 ms | +7.8%，量级复现 |
| **nn `--f16` ms/step** | **118.4** | **138.8** | **+17.2% ⚠ 退化** |
| **nn `--f16` 显存** | **2113 MiB**（nvidia-smi） | **6972 MiB**（PDH） | **+230% ⚠ 回归**（同轮 f32 两口径差 ~3%，解释不了；§4.3） |
| nn 全 f16 | 3 epoch 跑完、`-nan`、1526 MiB | **首步 OOM**、7365 MiB（崩溃前） | 都不可用，表现恶化 |
| f32 显存 nn / torch | 3141 / 3313 MiB（nvidia-smi） | 3237 / 3741 MiB（PDH） | 口径不同；同轮内 nn 仍更省（0.87×） |
| 低精度显存 nn / torch | 2113 / 1981（nn 多 7%） | 6972 / 1915（**nn 多 264%**） | **反转，见 §4.3** |
| GPU busy% | 89.8% | **缺失（待查）** | 无法复核 |
| 语料 | sha `584422376434996e…`（3,381,238 B） | sha `0a2edd42958cfc7b…`（3,377,345 B） | 重建；本轮两侧同用 |
| nn 版本 | `652c53d` + 未提交优化、AOT 融合 shader | `90dce64` 干净树、运行期 JIT + 引擎 DLL | 见 §1 |
| torch 版本 | 2.14.0+cu132 | 2.15.0+cu132 | 见 §3 |
| 环境异常 | （无记录） | NVML 失效 / WMI 名不符 / 驱动更新 | §1、§6.6 |

**不复现/变化项的处理**：与 10-06 相同——如实记录新值、保留旧值追溯，未对旧结论做"勘误"（f16 显存/速度与设备表缺失均未二分，标待查）。`docs/internal/14` 的全 f16 缺陷与 18 号 C9 的 f16 降速记录均**仍然成立且表现加重**。

---

## 6. 方法论与已知偏差

1. **同窗对照**：本轮 nn 与 torch 在同一会话、同一张卡、同一份语料上采集；与 10-06 之间的任何逐毫秒比较仍含跨轮漂移（torch 版本、驱动、语料、构建均不同）。
2. **计时口径**：两侧均为训练主循环 wall-clock，epoch 边界同步 GPU；epoch 内 host（构 batch / H2D）与 GPU 重叠，不做逐步 synchronize。稳态取 epoch 2–3 平均。
3. **轮次纪律**：3 epoch（与历轮一致）；epoch 1 只作 warmup 不进稳态；torch fp32 两进程复跑 304,534 / 303,464 tok/s（±0.35%），nn f32 复跑 150,059 / 145,966 tok/s（±2.7%）。
4. **词表差异**：C++ BBPE 8208 vs torch HF BPE 8192（本轮现场训练），两侧分词器独立、token 流不同，LM-head FLOPs 差 0.16%；速度按 tok/s 归一。
5. **掩码语义**：nn 训练启用 per-line 文档掩码（fold 内生效），torch 为标准 causal——FLOPs 相同，语义不同，速度无影响。
6. **显存口径变更**：本轮 `nvidia-smi` 因 NVML 初始化失败不可用，`run_with_vram.ps1` 自动回落 PDH `\GPU Adapter Memory(*)\Dedicated Usage`（取全部适配器最大值，等价于旧口径的 NVIDIA 整卡 `memory.used`）。**与 10-06 及更早的 nvidia-smi 数字跨口径，不可逐字比较**；同轮内两侧同口径可比。PDH 单次 `Get-Counter` 开销大于 nvidia-smi，实际采样间隔约 0.7–1.5 s（旧口径 ~150–300 ms），对快速瞬态的捕捉更弱——f32/f16 各步平台期远宽于该间隔，峰值可信；全 f16 仅 2 个采样点（进程数秒内崩溃），其 7365 MiB 为下界。
7. **profile 口径**：`NN_PROFILE=1` 默认关闭零开销；本轮 f32 20 步墙钟比 e2e 高 6.5%（116.3 vs 109.2 ms/step），整 epoch 口径 +3.8%。f16 剖面（+65%）含首 epoch 池增长期，只作结构/均值对照。设备侧 timestamp 表本轮缺失（§4.2）。
8. **本机噪声与环境**：40HX 为专用计算卡（显示输出走 AMD R5 240）；NVML 失效与 WMI 显示名差异（"RTX 2060 SUPER"）记录在案，Vulkan/CUDA 两侧枚举互证同卡；构建期间发现 HEAD 上 `precision_test` 宏重定义编译错（与基准无关，已绕开——按目标构建应用，不建测试）。

---

## 7. 产物索引与复现命令

**本轮原始数据（`build/`）**

- nn f32：`nn_short_f32_20261112.log`（+ `.vram` / `.peak`）、复跑 `nn_short_f32_r2_20261112.log`
- nn `--f16`：`nn_short_f16rec_20261112.log`（+ `.vram` / `.peak`）
- nn 全 f16：`nn_short_f16_20261112.log`（⚠ OOM）、逐步复现 `allf16_steps_20261112.log`
- torch：`torch_short_fp32_20261112.log` / `torch_short_fp16_20261112.log`（+ `.peak`），复跑 `torch_short_fp32_r2_20261112.log`，`torch_train_fp{32,16}_20261112.json` / `torch_train_fp32_r2_20261112.json`
- step 剖析：`prof_bench_20261112.txt`（f32 20 步 + 整 epoch 重写同名）、`prof_bench_f16_20261112.txt`、`trace_bench_20261112.json`
- 单步 probe：`probe_torch_step_20261112.log`；显存对照探针：`mem_probe.ps1` + `probe_f32.out` / `probe_f16.out`（`-Tag cmp02`）
- 数据集（重建）：`datasets/tinystories_bench40.txt`、`build/bench_bpe_8192.nnvocab`、`build/tinystories_bench40.nndataset`、`build/fetch_corpus.py`；torch 词表 `bench/raw/torch_bpe_8192.json`

**复现命令**（仓库根目录；PowerShell）：

```powershell
# 0) 重建语料（HF TinyStories train 前 4000 行，带断点续传）
.\.venv\Scripts\python.exe build\fetch_corpus.py
#    → datasets\tinystories_bench40.txt（sha256 0a2edd42…；源 URL/清洗见脚本头部）

# 1) 构建（本轮 MSVC STL 升级 19.51→19.52 后全量重编；应用目标不建测试）
cmake --preset release
Remove-Item build\release\modules\*.pcm -ErrorAction SilentlyContinue
ninja -C build\release text_train text_infer tokenizer_train tokenizer_infer dataset_gen nn_warmup layer_bench

# 2) nn 侧：词表 → 统一数据集 → 训练（engine DLL 在 PATH）
$env:PATH = "C:\...\neuralnet.cpp\build\release\engine;$env:PATH"
.\build\release\tokenizer_train.exe datasets\tinystories_bench40.txt --vocab-size 8192 --output build\bench_bpe_8192.nnvocab
.\build\release\dataset_gen.exe datasets\tinystories_bench40.txt --vocab build\bench_bpe_8192.nnvocab -o build\tinystories_bench40.nndataset

#    f32 主对比（含显存采样；NVML 不可用时脚本自动走 PDH）
.\bench\run_with_vram.ps1 -Exe .\build\release\text_train.exe -Log build\nn_short_f32.log -Arguments @(
  'build\tinystories_bench40.nndataset','--d-model','64','--num-heads','4','--num-layers','4',
  '--d-ff','256','--seq-len','256','--batch-size','64','--epochs','3',
  '--optimizer','adam','--lr','0.001','--gpu=40HX',
  '--save-interval','0','--save','build\gpt_bench_nn.nnmodel','--log-interval','100000')

#    --f16 推荐低精度：同上，追加 '--f16'
#    全 f16（⚠ 本轮首步 OOM）：追加
#      '--precision-param','f16','--precision-compute','f16','--precision-stable','f16','--precision-optimizer','f16'

# 3) torch 侧（同一语料；词表缺失时脚本现场训练）
$py = '.\.venv\Scripts\python.exe'
$common = @('bench\train_torch.py','datasets\tinystories_bench40.txt','--vocab','bench\raw\torch_bpe_8192.json','--vocab-size','8192','--epochs','3')
.\bench\run_with_vram.ps1 -Exe $py -Log build\torch_short_fp32.log -Arguments ($common + @('--dtype','fp32','--out','build\torch_train_fp32.json'))
.\bench\run_with_vram.ps1 -Exe $py -Log build\torch_short_fp16.log -Arguments ($common + @('--dtype','fp16','--adam-eps','1e-4','--out','build\torch_train_fp16.json'))

# 4) torch 单步分解（§3）
.\.venv\Scripts\python.exe bench\probe_torch_step.py

# 5) nn step 剖析（§4，与 e2e 同配置）
$env:NN_PROFILE='1'; $env:NN_PROFILE_TRACE='build\trace_bench.json'
.\build\release\text_train.exe build\tinystories_bench40.nndataset --d-model 64 --num-heads 4 --num-layers 4 `
  --d-ff 256 --seq-len 256 --batch-size 64 --epochs 1 --max-steps 20 `
  --optimizer adam --lr 0.001 --gpu=40HX --save-interval 0 `
  --save build\gpt_prof_nn.nnmodel --log-interval 100000 *> build\prof_bench.txt
Remove-Item Env:NN_PROFILE,Env:NN_PROFILE_TRACE

# 6) 显存分适配器对照（§4.3 探针）
.\build\mem_probe.ps1 -Tag cmp02
```

