# Model 级融合试点设计（P4-5）

> **读者**：项目开发者（改融合 / 引擎录制 / 训练路径的人）。本分册定位见 [`README.md`](README.md)；
> 历史归档在 [`../history/history.md`](../history/history.md)（裁定条 =「Model 级融合撤销否决并裁定试点」）。
> **一句话结论**：试点目标 = 首跑（或训练前手动一次）把 model forward/backward 捕获为模型级录制图并产出融合计划，此后训练步只执行该计划；当前为设计稿（结构障碍解法 + 分期），**未实施**。
> **状态**：TODO

## 1. 裁定与范围

- 裁定坐标：[`18-roadmap.md`](18-roadmap.md) §4.5 **P4-5**；执行排序见 [`27-execution-plan.md`](27-execution-plan.md) §7 第 3 条。
- **捕获模型（用户裁定）**：Model 级融合在**首次调用 model forward/backward 时自动运行捕获**，也可像现有表达式一样**训练前手动调用一次融合**；此后除非手动再录，**训练步执行的算子只有模型级融合计划**——不再逐表达式派发。
- **定位（评估结论，见 history 裁定条）**：f32 对拍差距中框架开销 ≈13%、设备算子 ≈87%；本项吃开销与表达式设备小头，**GEMM 主差距（61% 设备侧）由 P4-2 承接**，两者不互斥。
- **非目标**：① S6 自动窗口（邻接表达式无录制自动并入，仍触发式，18 §6）；② 替代 P4-2 的 kernel 效率调参；③ 试点期不动 optimizer（已融合，见 §2）与 CE loss（已 fold，扩展项见 §4 Phase 4）。

## 2. 现状机制（代码事实）

| 项 | 现状 | 坐标 |
|---|---|---|
| 录制 API | `ComputeEngine::begin_expr()` / `end_expr()` + `ExprSegment` RAII 收口 | `engine/abstraction/include/neuralnet.cpp/compute_engine.hpp` |
| 生产调用方 | **仅** `Adam::step` / `AdamW::step`；Model/Layer 不包录制 | `include/neuralnet.cpp/compute_optimizer.hpp:492,578` |
| 嵌套录制 | **禁止**（`begin_expr` 嵌套 = 报错） | `engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp` |
| 录制段内禁用 | `eval_expr_reduce_into` / `eval_expr_multi_into` 不支持在录制段内 | 同上 |
| 融合规则 | P1 链（matmul/fold 头 + 逐元素尾，tail 恰一消费者）；P2 分量写穿（纯逐元素 F32/F16，`extras ≤ 64`、链长 ≤ `EXPR_MAX_INSTRS=4096`、寄存器 ≤ `EXPR_MAX_REGS=255`）；P3 跨链批量（同结构组、CAP 32） | `engine/abstraction/include/neuralnet.cpp/expr_graph.hpp`、`expr_spec.hpp` |
| 图级计划缓存 | `graph_cache_key` / `FusedKernelPlan` / `instantiate_plan`（跨 step 复用） | `expr_graph.hpp` |
| CPU 语义 | `begin_expr`/`end_expr` 为 no-op，数值不变 | `engine/engine/include/neuralnet.cpp/compute_cpu_engine.hpp` |
| Model 结构 | `Model::forward` 逐层循环、`Model::backward` 倒序回放；层间无图 | `include/neuralnet.cpp/model_container.hpp:239,258` |
| 每步派发量 | ≈475 次表达式派发/步（`eval_expr` 184 + `eval_expr_into` 291） | `docs/benchmarks/2026-11-12-vulkan-vs-torch.md` §4.2 |
| 运行期 JIT | 结构 key → `.nnk` 缓存；复合结构未命中拆回逐成员（可观测，非错误）；`model_warmup()` 预热 | `docs/internal/33-kernel-cache-and-engine-decouple.md`、AGENTS §7 |

## 3. 关键设计问题与草案解法

> 本节均为**设计意图**（未实施）；每个 D 的最终取舍在试点实测后回写本文。

### D1 捕获范围与"单一表达式"的读法

- 捕获边界 = `Model::forward` 与 `Model::backward` 各一次录制段（loss / optimizer 不在试点边界内，理由见 §1 非目标）。
- **"模型级的单一表达式"落地为"模型级的一份计划"**：一次捕获 → `fuse_expr_graph` 融合分析 → 按预算切分的 kernel 序列 + 派发编排。预算侧限制已放大（`EXPR_MAX_INSTRS=4096`、`extras ≤ 64`、`EXPR_MAX_REGS=255`——后者即 `ExprOperand.idx` uint8 的 ABI 天花板），**剩余硬墙** = `EXPR_MAX_INPUTS=16`（`ExprPrecSig` 16 位输入位图，放大须重排签名位）、单 matmul 段（`spec.matmul` 单 `optional`，双 matmul 已 fail-fast）、归约同轴规则、push constant 设备上限——这些决定分段边界；计划在语义上替代逐表达式派发即可满足裁定。
- 分段策略：按依赖 DAG 拓扑序 + 预算（指令数 / extras / 输入槽 `EXPR_MAX_INPUTS`）切分；**段边界 = 需写穿到 HBM 的张量处**（段内中间量留寄存器/shared，段间落盘）。

### D2 backward 缓存逃逸（历史否决的核心点）

现状：各层为 backward 缓存的中间量（如 `LayerNorm` 的归一化结果、`GPTBlock` 的残差缓存）是**录制图之外的第二消费者**，融合分析看不见它。

- **方案 A（试点采用）**：捕获期把 backward 需要的中间量**显式建模为图输出**（P2 写穿的自然延伸）——缓存从"逃逸"变为"计划的正式输出槽"；写穿预算超限处即 §D1 的段边界。
- **方案 B（可选模式，后置）**：捕获期不缓存、backward 全量重算（等价 `checkpoint_every=1` 的极端档），以显存换融合率。与梯度检查点 / activation offload 的交互单列，试点期这两类特性启用时**回退逐层路径**（见 D8）。

### D3 录制段内的归约写入

`LayerNorm`/`RMSNorm` 层路径已按 P4-2 单算子化收拢：前向 2 段、LN backward 5 段（RMSNorm 2/3——统计量经链式归约指令收进单个 `dsl::compute`，`include/neuralnet.cpp/compute_layer_mlp.hpp`）；dγ/dβ 仍为独立 `compute_reduce` 行归约。

- 草案：① 归约节点作为**段收口点**（reduce 在段边界单独执行，段间经写穿衔接）；② 与 **P4-2 的「LN/RMSNorm 单算子化」子项同源**——该子项用**链式归约指令**（归约指令源可引用更早归约结果；`docs/internal/02` IR 扩展章）把 LN 前向收成单表达式后，fold 头/归约头 + 逐元素尾链已属 P1 允许域，本障碍随之消失。**注意：fold 不适合 LN 全网格输出**（fold 输出形状为 `(rows, out_cols)` 且 body 域禁归约视图——CE 适用是因为只需标量输出）；两条路线按 P4-2 先行结果择优。

### D4 计划缓存、结构 key 与预热

- 计划的结构 key 沿用图级 `graph_cache_key`（结构描述，不含形状/精度签名——与 `expr_spec_key` 同口径）；跨 step 经 `instantiate_plan` 复用。
- **捕获时机与 `model_warmup()` 合流**：warmup 本就跑一遍代表性 forward，捕获挂在其上即"首跑捕获"；手动入口（`Model`/训练入口暴露一次融合调用）与之等价。
- 结构随模型配置（层数 / 头数 / `norm_type` / 精度 profile）分桶；持久化复用 `.nnk` 机制，部署首跑语义不变。

### D5 与 checkpoint / offload / 精度的关系

- 试点范围 = 常规训练路径（`checkpoint_every=0`、offload 关、f32 与既有 profile）。捕获入口检测到梯度检查点或 activation offload 启用 → **不捕获，回退逐层路径**（记录一次可观测事件）。
- 混精度 profile：融合分组依赖精度（P2 需 F32/F16 存储成员），混精度下的未命中已有"逐成员降级"通用机制（AGENTS §7），模型级计划沿用同一降级哲学。

### D6 CPU 语义

CPU 上录制为 no-op ⇒ 捕获在 CPU = 直接执行、数值不变；模型级计划只在 GPU 生效。与 IR-C 现状一致，CPU 侧零风险、零收益。

### D7 确定性与字节锚

- 验收底线 = **融合成员输出与未融合路径逐位一致**（沿用 `expr_fuse_test` / P3 批量的逐位口径，扩展到模型级计划）。
- 若 D3 采纳 fold 化 LN 等改变归约顺序的路线 → 属数值算法变更，按字节锚流程处置（重测 + 更新 AGENTS 锚行 + history 记录）。

### D8 开关、降级与回退

- 运行期开关（行为开关，初始化读一次并缓存，不入探针编译门）：捕获/计划执行可整体关闭做同窗 A/B，类比 `NN_IRC_NO_BATCH`。
- **降级层级**：计划执行中结构未命中或前置条件不满足 → 拆回成员逐个派发（与 IR-C 复合降级同哲学，可观测非错误）；捕获本身失败（不支持的层 / 嵌套冲突 / 预算不可切分）→ 回退现行逐层路径。单节点结构未命中仍走铁律 7 的硬报错，不因本项放松。

## 4. 分期

| 阶段 | 内容 | 出口判据 |
|---|---|---|
| **Phase 0** | 基线：重跑 `docs/benchmarks/` 同任务对拍（27 §7 第 3 条既有要求） | f32 比值与 ms/step 基线入档 |
| **Phase 1** | forward 捕获：先 GPT 单 block（D1 分段跑通），再整模型 forward；含 D4 计划缓存接线与 D8 开关 | 成员输出逐位一致；单 block 派发数下降；开关回退可用 |
| **Phase 2** | backward 捕获（D2 方案 A：缓存显式写穿）；D5 检测回退落地 | backward 逐位一致；checkpoint/offload 启用时正确回退 |
| **Phase 3** | e2e 验收（训练路径，f32） | §5 全项 |
| **Phase 4（扩展，不阻塞试点）** | CE loss 并入捕获边界；D3 路线②（LN fold 化）与 P4-2 汇合；方案 B 重算模式 | 另立验收 |

## 5. 验收口径（Phase 3）

1. **派发/kernel 数**：每步表达式派发数相对基线 ≈475 **显著下降**（目标值在 Phase 1 实测后回填本文）；`NN_PROFILE` 设备 kernel 表同步下降。
2. **e2e**：同窗配对 A/B，ms/step 下降且 loss 曲线与基线重合（3 epoch 对拍口径）。
3. **正确性**：融合成员输出与未融合路径逐位一致；既有 ctest 融合/训练类用例全绿。
4. **torch 对照**：按 18 §4.7 增加同任务 torch 口径（比值变化如实入 `docs/benchmarks/`）。
5. **回退**：D8 开关关闭时行为与现行路径一致（A/B 各跑通）。

## 6. 风险

- **收益不及预期**：开销占比 ≈13% 决定了本项 e2e 上限（估算 ≤15%）；若 Phase 3 实测低于预期，按 18 P4-5 行的定位降级为"开销项"继续留在 P4，不挤占 P4-2 主杠杆。
- **结构组合膨胀**：计划结构随配置分桶，首跑编译量与 `.nnk` 体积增长——由 warmup 与既有缓存机制承接，Phase 1 观测 miss 率。
- **工程面大**：D2/D3 触及录制语义与层缓存协议，必须分阶段出口判据全绿再进下一阶段（§4），不允许跨阶段并进。
