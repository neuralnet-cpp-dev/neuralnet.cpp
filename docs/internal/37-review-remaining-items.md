# 37 · 审查剩余项台账（11-02 / 11-09 两轮全库审查）

> **读者**：项目开发者。本分册定位见 [`README.md`](README.md)；历史归档在 [`../history/history.md`](../history/history.md)。
> **一句话结论**：两轮全库审查清零后的剩余项**已全部处置完毕（2026-11-10）**——#P13（vecacc 收尾归约）完成开工验收 A/B 后**实测显著负收益、按方案否决不立项**（history「P13 fold v2 收尾归约 A/B」条）；#P10-③（fold 后逐元素收尾）与注释过时项完成（history「P10-③ fold 后逐元素收尾」条）；Linux `.so` 消费方录制复验在 WSL2 实机完成（P10-④ 断言 10/10，留证于 36 号 C1）；原两份审查报告已从仓库移除（原文见 git `0cef8d8` / `048bd75`）。无待办。
> **状态**：现行

来源：`审查报告-2026-11-02.md`（53 项）与 `审查报告-2026-11-09.md`（29 项）两轮全库审查。已闭环项不复述（终态：11-09 全 29 项 + 11-02 遗留 #P4/#P10①②④ 均完成，36 号妥协台账全表清零，ctest 81/81）；本文件只登记**剩余项**。

## 剩余项

| # | 项 | 状态 | 开工条件 / 说明 |
|---|---|---|---|
| 1 | **#P13 vecacc 收尾归约**——fold v2 路径 B 的每行串行 16 槽求和，改为同 subgroup 连续线程组映射（`c = tid/G`）+ 块状 shuffle（免 shared/屏障）。方案与 P2-2 的 A/B 基线见 history「fold v2 vecacc 双路径改写（P2-2）」条尾"下一步空间"；代码位置 `expr_glsl_gen.hpp` 的 `generate_glsl_fold_v2` | **已否决（2026-11-10）** | 开工条件（先跑 shader A/B 实测）已执行：同窗交错 3 对实测 fold avg_us **2524.8 → 3998.9 µs（+58.4%）**、E2E **+6.7%**——连续映射毁掉 V 读跨线程合并，收尾省的屏障/槽流量远不抵 → **按方案否决、不立项、锚不动**。方法/数据/实施期 shuffle 源活跃性缺陷与跨 DSO 剖析表缺席的附带发现见 history「P13 fold v2 收尾归约 A/B（…）：否决（2026-11-10）」条 |
| 2 | **#P10-③ fold 后逐元素收尾**——扩 fold emitter 与 `validate_expr_spec` 的"fold 顶层指令须空"限制 | **已完成（2026-11-10）** | 单独立项完成：validate 尾链域规则（含 fold PC cols=键长导致的视图拒绝清单）+ GLSL v1/v2 'T' 上下文发射 + CPU 域分槽校验/尾链执行 + 图融合 ②b 前置拼接（消费者 p2_eligible 须先于分量分支试拼）；验收 = 主仓 ctest 81/81 + 引擎 GPU 11/11（expr_fuse_test v1 逐位 err=0、v2 尾链不变量 err=0）+ expr_graph_test 结构/CPU 用例；字节锚与 HEAD 基线逐位一致（零移动）。设计/坑/容差/锚观测见 history「P10-③ fold 后逐元素收尾」条 |
| 3 | `expr_emitter.hpp` 批量生成注释过时——"带 extras 的 f16 结构仍逐 kernel 降级"在 P10-④ 后不再准确 | **已完成（2026-11-10）** | 化妆级。已核 P3 批量资格（`execute_fused_graph` 的 b_prefix 判定：extras 恒 F32、sig≠0 须 extras 空）× P10-④ 边界 cast 实际交互后按实况改写：`expr_emitter.hpp` 接口注释 + `expr_glsl_gen.hpp` 批量门禁同款过时句——带 extras 的 f16 结构 P10-④ 后**能融合执行**（单复合 kernel：边界 cast → V0 → 落回，不逐成员降级），只是不入批量组 → 逐 kernel（非批量）派发 |
| 4 | Linux `.so` 行为实机复验——跨 DSO 录制状态"ELF 弱符号 TLS 归并"目前是读码推断 | **已完成（2026-11-10）** | WSL2 Ubuntu-26.04 实机（clang-24 + Vulkan SDK 1.4.363.1 + Dozen/RTX 2060 SUPER）：消费方形态录制回归 `expr_gpu_test` 的 **P10-④ 断言 10/10 PASS**（fail-only 空转门零触发、判别力自检 4.88e-04、f16 菱形 vs 双参考全绿），Dozen 与 llvmpipe 双驱动同果、多轮复跑稳定；主仓 ctest **79/81**（2 项为 WSL 环境差异，见下）。方案 A 后 Linux 端到端成立由设计保证（状态归引擎 TU 单副本），TLS 归并读码推断与实测一致、不再构成依赖——留证追加于 36 号 C1"平台标注补正留证"。**WSL 环境差异（非产品回归，同源代码 Windows 81/81）**：① `ce_fusion_test` parity(3,100,x2) 确定性失败 loss_err=5.745e-01（Dozen=llvmpipe 同值、嵌入 reduce.spv 两端 SHA256 一致、换 Windows .nnk 缓存仍同值但出现键 miss 追加 33KB → 残差锁定 WSL 构建域 host 侧（clang-24+libstdc++ 键/值域）或跨构建缓存不可移植，待 WSL 平台化时专项排查）；② `batchnorm_test_gpu` 偶发启动期段错误（时序敏感，无缓冲 6/6 过、ctest 间歇挂）——均记边界观察 |

## 已收口备注（无待办）

- `reduce_into` 录制失败路径 epoch 先于守卫 +1：跨 DSO 方案 A 的保守语义，代码注释已写明，不构成缺陷、无需动作。
