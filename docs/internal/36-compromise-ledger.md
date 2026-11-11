# 36 · 模块迁移妥协式设计台账（最终验收前必须清零）

> **读者**：35 号的实施者与验收者。
> **一句话结论**：为解锁验收门引入的妥协式绕行逐条登记，**门 G 前必须逐条消灭、不得转正**；根因级修复不入表。
> **状态**：现行（随 35 号实施过程滚动登记）。
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
| C1 | `include/neuralnet.cpp/compute_optimizer.hpp`：Adam::step / AdamW::step 的 `graph_ok = precision_bytes(p_.optimizer) >= 4` 分支 | optimizer=f16 时**跳过 IR-C 录制退直求值**：full-f16 图路径第 2 步即 NaN，根因（"录制 × f16 状态"组合）未下钻即绕行；该分支使 full-f16 永久失去图融合 | ✅ **已消灭（2026-11-07）**。**根因定案（与设备无关，先前"设备差异"收窄被推翻）**：全 f16 图下 K2（v 更新）与 K3/p+ 同精度融合为**单输出** kernel（P2 写穿 extras 只允许全 F32），`v_[i] = move(*v_new)` 持有的占位被内联为融合中间量**从不写入** → 下一步读内存池未初始化残留（本机恰为负值）→ `sqrt(v)` → NaN。四层对拍（输入位级 → to_f32_all → f32 eval → cast_into）证明边界 cast 链全程净；122 个 JIT 缓存 kernel 零 Float16 capability；"他机不复现"= 池残留内容环境依赖（40HX 恰为 0）。**修法三件套**：① K1/K2 改 `compute_into`（into = 落盘声明）；② 融合器拒绝把 `dst_id != 0` 节点拼为非尾（`[ir-c][fuse-decline]` 降级打印）；③ into 输出 stamp tag 恢复图内依赖（保住 f32 四成员 extras=3 融合）。**门控已删**（`graph_ok` 恒开图），教训入 `docs/contributors/pitfalls.md` §2.6 + 模式 I。留证：850M 锁定 `f16_precision_test` ALL PASSED（full-f16 常态走图）、`ctest --preset dev` 81/81 | 最终清理（P6 后、门 G 前） | ✅ |
| C1-调查 | **2026-11-06 复现与排除进展**：① 复现稳定——`test_full_f16_training` 临时强制 `graph_ok=true` 后 GPU（850M）step 1 更新即把全部参数打 NaN（loss 3.4778→3.4626→-nan；CPU 直求值同 profile 全绿）。② 排除项：NN_NO_GRAPH=1（GPU 直求值）全绿 ⇒ 根因在**图执行路径**；禁用 graph_plan_cache（`NN_IRC_NO_PLAN_CACHE` A/B）仍 NaN ⇒ 非 P2-12 图级缓存；cast.comp（F32→F16 RNE 写回）审查无误。③ 现场：full-f16 图 = 152 个全 F16 节点（P0=152），fuse 拆成单输出 kernel（P2 写穿要求全 F32 ⇒ 无 extras 复合），全部走 `run_graph_kernel_into_` 的 psig≠0 **边界 cast**（C4 路径）；K1（m 更新，key fd3bd576）/K2（v 更新，key a3baa338）/K3（delta）的边界 cast 输入出现 NaN（m/v 在 adam_update_ 录制前下载检查**正常**——0→1.32→2.38 / 0→0.17→0.32），且某 step 的 v_[i] 读到**负值 -2**（v=β2v+(1-β2)g² 数学上不可能负）。④ **关键谜题（未解）**：K1 录制占位 P=0（F16，record_graph_node_ 打印确认），但**同一 step 的 K3 录制时 m_[i]/v_[i] 的 `precision()` 报告 F32**（K3-record 打印），dispatch 时 d.ins/d.out 也全 F32——**占位张量的精度状态与缓冲不一致**（f16 缓冲 + f32 精度报告，或 m_[i] 句柄在 K1→K3 之间被改），怀疑与「抽象层头（主程序 TU 实例化 NVI）与引擎 .so（GpuEngine 实现）跨 DSO 的实体/精度视图分裂」有关（**平台标注补正 2026-11-10：`recording_graph_owner` 的 inline thread_local「已实测共享」仅在 850M/Linux .so 为真——ELF 弱符号 TLS 归并；Windows/DLL 为假**——PE 模块 TLS 独立，NVI 录制检查读消费方 EXE 副本恒空 → begin/end 虽在 DLL 侧配对成功但图 nodes=0、占位 tag=0、**GPU 录制空转**；本专项读码+消费方形态实测坐实，**修复 = 方案 A（录制状态读写全部归引擎 TU**：begin/end + 两引擎 `eval*_impl` 顶部拦截，NVI 零状态读），回归锁 = `src/fused_gpu_test.cpp` P10-④ tag≠0 FAIL 断言；同括号内「张量精度状态路径未查完」的子谜题随后由本行⑦定案）。⑤ 本轮未修（深水区，需专门会话继续——建议下一步：打印 K3 录制时 m_[i] 的 `virtual_tag` 与 K1 占位 tag 是否一致，确认 m_[i] 句柄是否真的指向 K1 占位）；**C2/C4 依赖 C1，同组卡住**。⑥ **2026-11-07 40HX（CMP 40HX）复现**：graph_ok 恒 true + `NN_VULKAN_DEVICE` 锁 CMP 40HX，同代码 `f16_precision_test` **ALL PASSED**（EXIT=0，full-f16 GPU 训练 loss 有限下降，无 NaN）——「占位张量精度状态/跨 DSO 视图分裂」怀疑在 40HX 上不成立或无害；850M 的 NaN 收窄为 **Maxwell 无 shaderFloat16 的 f16 回退/边界 cast 写回路径设备问题**。注：40HX 复现环境需完整 Windows 构建修复链（NOMINMAX / _CRT_SECURE_NO_WARNINGS / BMI CRT 宏镜像 / WINDOWS_EXPORT_ALL_SYMBOLS WIN32 / shaderc HINTS / kernelgen dll 平台名与导出 / 测试文件显式 STL include / .nnk 缓存目录），见 AGENTS 待更新清单。⑦ **根因定案（2026-11-07 晚，推翻④⑥的两个方向）**：④ 的"精度状态分裂"与⑥的"设备 f16 回退"**均非根因**——四层对拍（L0 输入位级直读全净 → L1 to_f32_all → L2 f32 eval 突变产出 NaN → L3 写回）+ 按段融合形态对照表（f32/param=f16/opt=f16 段 extras 落盘全绿 vs all=f16/fullf16 段 K2+K3+p+ 三成员融合 extras=0）钉死：**K2（v 更新）输出被单输出融合内联、占位从不写入，`v_[i]` 读未初始化池残留（负值垃圾）→ `sqrt(v)` = NaN**；v=β2v+(1-β2)g² 数学恒非负 ⇒ ③ 的"v 读到负值 -2"即此现象。定位手段与教训沉淀见 `docs/contributors/pitfalls.md` §2.6（灾难级）。⑤ 的"深水区"就此关闭 | 同 C1 | ✅ |
| C2 | `engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp`：begin_expr / end_expr 的 `NN_NO_GRAPH` getenv 旁路 | A/B 排查遗留：整跳录制的逃生阀，代码注释自标"临时" | ✅ **已消灭（2026-11-07）**：两处 getenv 分支已删（与 C1 同组）；排查使命随根因定案结束，环境变量无任何残留引用（全仓 grep 清零）。验收 = 850M `f16_precision_test` ALL PASSED + `ctest --preset dev` 81/81（full-f16 常态走图、无旁路） | 同 C1（一组） | ✅ |
| C3 | `engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp`：execute_fused_graph 的 `[ir-c]` 打印 | ① degrade 分支由 `nn::dsl::env_flag` 改成裸 `getenv`、② 新增每 kernel 的 `[ir-c][run]` 打印——均属排查期改道；且 env_flag 受 NN_ENABLE_PROBES 门控使 AGENTS 承诺的 `NN_IRC_TRACE=1` 实际不可用，属**未登记的行为口径变更** | ✅ **已消灭（2026-11-06）**：方案 ①——`NN_IRC_TRACE` 正式化为**非探针门行为开关**（裸 getenv 保留，AGENTS §7 注明）；**删除 `[ir-c][run]` 每 kernel 打印**（命中执行是热路径常态，刷屏无排查价值）；降级 `[ir-c][degrade]` 与批量 `[ir-c][batch]`/`[batch-degrade]` 打印保留（低频、可观测"某段悄悄没融合"） | 最终清理 | ✅ |
| C4 | `engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp`：run_graph_kernel_into_ 的 psig≠0 分支 | 图内 f16 统一**边界 cast**、弃用静态签名 JIT 直写——当时把 full-f16 NaN 归因于"静态签名 JIT"但**未经单变量实验证实**（后证实是 C1 组合）；性能上每次 f16 图成员多付临时分配 + 双 cast。期间还留有一次 A/B 未回滚的形态改写（override+tmp 递归 → eval_expr_impl 分配 + cast_into） | ✅ **已裁决转正（2026-11-07）**：C1 根因定案（融合吞持久输出）证明 NaN 与边界 cast 无关——四层对拍边界 cast 链全程净，"错误归因"解除。**形态改写升格为有据设计**（`eval_expr_impl 分配 → cast_into 写回` 与直求值同形态，注释已从排查口径改为设计说明）；边界 cast 保留为 f16 图执行的正确路径。**「静态签名 JIT 直写 vs 边界 cast」性能 A/B 不再是本条义务**（C1 债已清），如要优化 f16 图成员的双 cast 开销可另行立项 | 最终清理（依赖 C1） | ✅ |
| C5 | `engine/engine/src/kernelgen.cpp`：失败路径诊断 fprintf（"GLSL 生成为空" / "shaderc 编译失败"） | 排查期新增的输出行为；非掩盖缺陷，但属未评审新增 | ✅ **已消灭（2026-11-06）**：**显式裁决转正**——保留两处 fprintf + 文件头补设计注记（调用方只见 NNK_COMPILE_FAIL，缺此区分不了"生成器不支持形态"与"shaderc 报错"两类根因；失败路径非热路径无性能代价） | 最终清理 | ✅ |
| C6 | `src/tokenizer_consistency_test.cpp`：`old_pre_tokenize` 的 `markers` 特意用 `std::vector<std::string_view>`（库侧为 `std::string`） | **clang 24 上游 bug 的规避绕行**（P5 实测，2026-11-06）：同一 TU 存在两个 **identical 构造**的 `static` local（测试参考实现与库 `pre_tokenize` 的 markers 含相同字面量+相同 sort lambda）**且 import 面超阈值**时，clang 产出错误代码——同函数内相邻循环一个正常执行、另一个 0 次迭代（行为随布局漂移，探针互换），-O0/-O3 均复现。已排除：写法 UB（cppreference range-for/compare 前置条件均满足）、gcc 15.3 全过、clang 纯代码全过、LTO、双实体（nm 单份符号）、布局重合。触发矩阵：纯代码 0 import 过 / 4 链式 import 过 / +expr、+core 过 / **+cli 或 8 域组合或全 13 挂**；old 去 sort 或改名不解除、**改 string_view 解除**。修复=改型打破同形（对拍语义不变），根因在上游未收敛 | ① **已上报 LLVM issues #239625（2026-11-06）**：复测新 clang（++20261105085444）**仍复现**；**2026-11-07 补测 clang 23.1.6（++20261103094334）同型亦复现**（症状互换：库侧 `manual` 标记循环失效、测试侧 `old` 正常——两侧代码未动仅恢复同型即触发）——规避的充要条件是**两侧异型**，与具体类型无关（两侧同改 string_view 亦必同型炸）；测试侧 `string_view` 本就更优（字面量只读表、零拷贝零所有权），该差异兼作规避、非刻意降级；**2026-11-07 最小化（推翻早前"含 cli/8 域"的触发面记载）**：消费方 import 面裁到**单模块 `import nn.tokenizer`** 仍复现（core 经 export import 链传递可见）；测试文件再裁到 218 行对拍核心亦复现 → 最小复现 = 单 import + 一行同型 + 两处 identical 构造，材料打包于仓库外 `~/codes/c6-repro/`（README + 源码摘录 + trigger/normal 同源二进制对照），供外部 AI 审查"是否本仓 UB/非标"与 LLVM #239625 互证；消融进一步收窄——触发 =「import 面含 nn.cli（裁到 text_train_config+text_train_loop 两头发仍复现，单头不复现）× 模块导出 inline 函数（pre_tokenize）实例化 × 消费方本地 identical 构造」缺一不可；消费方去 regex 仍复现；自包含最小化（模拟 loop 静态函数/大导出面/简化 pre_tokenize）均不复现；repro 依赖项目真实 pcm（issue 附克隆+构建命令）。② **已过时并按当前事实更新（2026-11-10）**：库侧触发条件已随 **commit `f4c42ef`** 根治——`pre_tokenize` 的 markers 改**非 static constexpr** `std::array<std::string_view,14>`（函数内 static 局部量不复存在 →「同 TU 两个 identical 构造 static local」配对瓦解、零全局实体，clang 双副本发射无从发生；llvm#239625 同族根治，ctest 81/81 留证）——**「门 G 前撤 string_view 恢复同型」的旧义务随之解除（表述已删）**。测试侧异型**按现状保留**（`static const std::vector<std::string_view>` vs 库侧 constexpr array），原因：① 触发配对已不存在，异型不再是规避的必要条件；② 只读表用 `string_view` 本就更优（零拷贝零所有权，原判断仍成立）；③ 上游 #239625 关闭前兼作防御（若有人把两侧改回同型 static 形态仍有回旋余地）。非待办 | 已随 `f4c42ef` 完成；上游 #239625 关闭仅作记录 | ✅ 已消灭：`f4c42ef` |

## 消灭顺序

**进度（2026-11-10）**：C1+C2 ✅（根因定案 + 三件套修复，门控/旁路已删）→ C4 ✅（裁决转正）→ C3 ✅ → C5 ✅ → C6 ✅（`f4c42ef` 库侧非 static constexpr 根治，测试侧异型保留为无义务现状）→ **全表已消灭（0 未消灭行）**。
C1/C2 验收留证（2026-11-07，已完成）：`ctest --preset dev` **81/81 全绿** + 850M 锁定
`f16_precision_test` **ALL PASSED**（full-f16 用例常态走图、无 NN_NO_GRAPH 旁路）；
`NN_IRC_TRACE=1` 下 `[ir-c][fuse-decline]` 降级打印实测生效。
C1 平台标注补正留证（2026-11-10，跨 DSO 录制视图专项）：Windows/DLL 录制空转
（tag=0/nodes=0 实测红）→ 方案 A（状态读写归引擎 TU）修复后 `expr_gpu_test`
P10-④ 断言全绿、engine-verify 4/4、dev-tests 全量 **81/81**、text_train 短跑
`NN_IRC_TRACE=1` 出现 `[ir-c]` 录制/融合输出。
Linux 侧补正（2026-11-10 晚，WSL2 Ubuntu-26.04 实机，37 号 #4 收口）：原"ELF
弱符号 TLS 归并"为读码推断——现以消费方形态录制回归实证：`libnn-gpu.so`（ELF）
+ 消费方 EXE 下 `expr_gpu_test` 的 P10-④ 断言 **10/10 PASS**（录制空转门 fail-only
零触发、判别力自检 4.88e-04>1e-6、f16 菱形 vs 双参考全绿），Dozen（RTX 2060
SUPER）与 llvmpipe 双驱动同果、多轮复跑稳定 ⇒ Linux .so 消费方录制端到端成立；
方案 A 后该成立由设计保证（状态全在引擎 TU 单副本），TLS 归并仅是 ELF 实现巧合
（与本行平台标注一致、不再构成依赖）。
