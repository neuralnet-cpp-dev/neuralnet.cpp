# 36 · 模块迁移妥协式设计台账（最终验收前必须清零）

> 状态：**现行（随 35 号实施过程滚动登记）**。目标读者：35 号的实施者与验收者。
> 关联文档：`35-modules-migration-implementation.md`（实施文档本体，只读）。
> **规则（用户裁定）**：实施过程中为解锁验收门而引入的妥协式绕行**一律登记在本表**；
> 允许阶段性带着它们推进（换取主线进度），但**最终验收（35 §6 门 G）的前提 =
> 本表逐条消灭**——不得转正、不得保留。每消灭一条在状态列记「✅ 已消灭：<commit>」；
> **门 G 验收时本表不得存在未消灭行**。
>
> 判别口径：**根因级修复不是妥协**（如 f16 缓冲 4B 对齐、`composite_kernel_available_`
> 无注册表误判、`cast_into` 奇数尾、BMI 命令镜像、四入口 JIT 兜底本体）——不入本表；
> **凡"根因未收敛就绕开 / A/B 实验残留 / 未评审的行为变更"必须入表**。

## 台账

| # | 位置 | 妥协内容（为什么是妥协） | 消灭方案 | 消灭时机 | 状态 |
|---|------|--------------------------|----------|----------|------|
| C1 | `include/neuralnet.cpp/compute_optimizer.hpp`：Adam::step / AdamW::step 的 `graph_ok = precision_bytes(p_.optimizer) >= 4` 分支 | optimizer=f16 时**跳过 IR-C 录制退直求值**：full-f16 图路径第 2 步即 NaN，根因（"录制 × f16 状态"组合）未下钻即绕行；该分支使 full-f16 永久失去图融合 | 下钻根因（重点：降级成员派发中 f16 状态 m/v 链的执行序、占位张量跨 step 复用、cast 写回路径），修复后**删除 graph_ok 分支恢复无条件录制**；full-f16 以「图开启 + ctest 绿」回归锁验证 | 最终清理（P6 后、门 G 前） | ⏳ |
| C2 | `engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp`：begin_expr / end_expr 的 `NN_NO_GRAPH` getenv 旁路 | A/B 排查遗留：整跳录制的逃生阀，代码注释自标"临时" | 删除两处 getenv 分支；与 C1 同组验收（删后 full-f16 仍绿） | 同 C1（一组） | ⏳ |
| C3 | `engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp`：execute_fused_graph 的 `[ir-c]` 打印 | ① degrade 分支由 `nn::dsl::env_flag` 改成裸 `getenv`、② 新增每 kernel 的 `[ir-c][run]` 打印——均属排查期改道；且 env_flag 受 NN_ENABLE_PROBES 门控使 AGENTS 承诺的 `NN_IRC_TRACE=1` 实际不可用，属**未登记的行为口径变更** | 二选一并同步文档：① NN_IRC_TRACE 正式化为非探针门的行为开关（更新 AGENTS/33 号口径）或 ② 恢复 env_flag 并改 AGENTS 注明需 NN_ENABLE_PROBES；**删除 [ir-c][run] 每 kernel 打印** | 最终清理 | ⏳ |
| C4 | `engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp`：run_graph_kernel_into_ 的 psig≠0 分支 | 图内 f16 统一**边界 cast**、弃用静态签名 JIT 直写——当时把 full-f16 NaN 归因于"静态签名 JIT"但**未经单变量实验证实**（后证实是 C1 组合）；性能上每次 f16 图成员多付临时分配 + 双 cast。期间还留有一次 A/B 未回滚的形态改写（override+tmp 递归 → eval_expr_impl 分配 + cast_into） | C1/C2 消灭后重新 A/B：静态签名 JIT vs 边界 cast，以 f16 基准**择优转正**（优则恢复直写；劣则把边界 cast 升格为有据设计并删实验残留形态） | 最终清理（依赖 C1） | ⏳ |
| C5 | `engine/engine/src/kernelgen.cpp`：失败路径诊断 fprintf（"GLSL 生成为空" / "shaderc 编译失败"） | 排查期新增的输出行为；非掩盖缺陷，但属未评审新增 | **显式裁决**：转正（补设计注记）或删除——不得默默留下 | 最终清理 | ⏳ |

## 消灭顺序

C1+C2 同组（互为 A/B 条件）→ C4 重新实验（依赖 C1 结果）→ C3 → C5。
消灭 C1/C2 后必须重跑：`ctest --preset dev` 全绿 + `f16_precision_test` 的
full-f16 用例（图开启、无 NN_NO_GRAPH）逐条留证。
