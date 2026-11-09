# 35 · src/ 全量 import 化与 AOT 收集 → JIT 预热 实施文档

> 状态：**方案已评审，待执行**。目标读者：实施者（人或 agent）。本文档自包含，可脱离会话上下文直接照做。
> 关联文档：`research/modules-import-nn/README.md`（研究与证伪记录）、`docs/internal/33-kernel-cache-and-engine-decouple.md`（JIT/缓存设计）、`docs/development/01-cmake-integration.md`（模块接入）、`AGENTS.md`（能力总览与铁律）。

---

## 0. 背景与根因

### 0.1 直接诱因：.pcm 编译选项不一致（config mismatch）

```
error: POSIX thread support was disabled in precompiled file '.../nn.tokenizer.pcm'
       but is currently enabled
error: module file '.../nn.tokenizer.pcm' cannot be loaded due to a configuration
       mismatch with the current compilation [-Wmodule-file-config-mismatch]
```

根因：BMI（`.pcm`）由 `nn_add_module` 内的**手写自定义命令**编译（`CMakeLists.txt` L537-L591），不链接任何 target；消费方经 `nn_core` 的 `Threads::Threads`（L351，`THREADS_PREFER_PTHREAD_FLAG ON`，L255-L256）拿到 `-pthread`。两边选项不同 → PCM 加载被拒。后续 `api-level.h`、`sys/cdefs.h` 等头文件的 `unknown type name 'define'`、`expected ';'` 全是**级联误报**（PCM 拒载破坏预处理器状态），修好首错即全部消失——**不得去改 NDK/libc++ 头**。

### 0.2 已核实的选项差异全表（P0 的靶子）

| # | 差异项 | 消费方 | BMI `_flags`（L552-L562） | 危害 |
|---|--------|--------|---------------------------|------|
| 1 | `-pthread` | 有（Threads::Threads） | 无 | **config mismatch（本次报错）** |
| 2 | `${CMAKE_CXX_FLAGS}`（调用方/工具链：NDK `--target=`、`-stdlib=` 等；L7 注明由调用方指定） | 有（CMake 自动拼） | **完全没拼** | 工具链/stdlib 不一致 |
| 3 | `${CMAKE_CXX_FLAGS_<CONFIG>}`（Release `-DNDEBUG`、Debug `-g`） | 有 | 无 | `assert()` 双语义 → ODR 风险 |
| 4 | `-DNN_GPU_ENGINE=1`（GPU 构建时 L1192 挂 `nn_core` INTERFACE） | 有 | 无（模块注册 L615 早于 `add_subdirectory(engine)` L1178） | 宏语义分叉 |
| 5 | `-I` 目录 | `nn_core` 5 个（L345-L350，含 `engine/engine/include`×2） | 只有 3 个 | 头解析路径不一致 |
| 6 | `-flto=thin`（L145） | 有 | 无 | 不触发本次报错（`--precompile` 无 codegen），但应同源 |

已确认**一致**的：`-std=c++26`（`CMAKE_CXX_EXTENSIONS OFF` + `cxx_std_26`）、`-fno-exceptions`、`-fexperimental-library`、`-Wno-pass-failed`、`-Wall/-Wextra/-Wpedantic/-Werror`、优化档（`${DEBUG/RELEASE_FLAGS}` 含 `-march=native`）、`-DNN_ENABLE_PROBES`。`NN_USE_MODULES` 宏**刻意不加**给 BMI（任何头都不读它，已全仓 grep 证实；模块单元源码永远走 `#include` 分支）。

---

## 1. 目标与范围

### 1.1 目标（用户定调，三条）

1. **修复编译选项不统一**（第 0.2 节全表归零）。
2. **`src/` 目录源代码不再出现 `#include <neuralnet.cpp/...>`，一律 `import nn.<域名>`**；覆盖 107 个 `.cpp` 与本地头 `src/expr_fuse_shapes.hpp`。
3. **不保留 `NN_ENABLE_MODULES` 开关、不做任何编译器识别**——其他支持 C++26 的编译器早就支持 modules，不支持的直接编译报错。

附带（用户提议、已纳入设计）：**删除 AOT 表达式收集链（`NN_EXPR_SCAN`），改为 JIT 预热**（运行期 `model_warmup()` 接线 + 构建期 `nn_warmup` 预热目标），借此解锁 engine/layers/model/train 域的模块化。

### 1.2 范围外（明确不改）

- `tests/`、`tools/`、`bench/`、`examples/`（含 downstream 独立工程）：继续 `#include <neuralnet.cpp/nn.hpp>`，门面 `neuralnet::nn` 不动。
- 库头文件内部的宏用法（`NN_TRY` 密集出现在 `compute_layer_rapt.hpp` 等）：头只被模块 GMF 文本包含，宏在模块 TU 内部合法，**零改动**。
- `CMAKE_CXX_SCAN_FOR_MODULES OFF`（L58-L67）与显式 BMI 自定义命令：维持（约 200 个非模块 TU 零扫描的成本决策）。
- `NN_TEST_COUNTER` 钩子（22 个测试文件 `#define` 后 include `src/test_common.hpp`）：本地头文本 include，不受禁令影响。

---

## 2. 关键既定事实（实施依据，均已核实）

### 2.1 JIT / AOT 现状

- `model_warmup(Model&, const Tensor&)` 已实现：`include/neuralnet.cpp/model_container.hpp` L350-L362——跑一遍代表性 forward，把用到的表达式现编进 `.nnk` 缓存。**当前 `src/` 无任何调用方**。
- JIT 架构设计见 `docs/internal/33` L39-L59；L125-L131 明确「`model_warmup()` 可拆掉 scan_exprs 的 FusedAnchor 自登记、dry-run、模型 pass、`NN_EXPR_SCAN`、glslc 构建步和 `nn_enable_gpu_fusion` 收集器；首次预热编译，之后共享缓存秒命中」。
- JIT 正确性已有回归：`src/fused_gpu_test.cpp` L563-L603（未注册表达式 → `libnn-kernelgen.so` 现编 → 执行 → 持久化，与 CPU 结果一致，`--hash` 位级对拍）。
- GPU 路径描述（AGENTS.md L483-L485）：`dsl::compute*` → 折成 `ExprSpec` 查构建期注册表 → miss 默认运行期 JIT → **`fold / reduce / into / multi` 入口 miss 硬报错** → 运行期 JIT 兜底（无严格模式开关）。
- CPU 路径模板直接求值，不依赖注册表（`docs/internal/02` L171：CPU no-op）。
- 收集器 `scan_exprs.cpp` 在源文件内 include 前 `#define NN_EXPR_SCAN`，不套 PCH、不进模块（L427-L430）；**不在 `src/` 下**（`src/` 仅 `expr_fuse_shapes.hpp:7` 注释提及）。

### 2.2 宏（模块化的关键约束）

- 宏不可随模块导出（C++ 标准）：`modules/nn.core.cpp` L5-L7 明注。
- `core_errors.hpp` / `core_assert.hpp` 位于 submodule `engine/abstraction/include/neuralnet.cpp/`（L487-L490 的 `NN_ABST_INC`）。
- 语义（依 `docs/development/02` L86-L122）：
  - `NN_TRY(expr)` / `NN_TRY(decl, expr)`：失败 **从当前函数 early-return** `std::unexpected`（带值传播）——**函数无法替代**；
  - `NN_TRY_MSG(decl, "ctx", expr)`：同上带语境；`NN_TRY_CHECK(x)`：检查已声明变量（副作用表达式会重复求值）；
  - `NN_UNWRAP(expr[, "ctx"])`：失败打印 file:line 后 abort，成功返回解包值（可函数化）；
  - `NN_EXIT(expr, code[, "ctx"])`：失败打印后 `exit(code)`（可函数化；测试用 77=SKIP）；
  - `NN_FAIL("msg")` = `return std::unexpected(Error{...})`；`NN_ASSERT` Debug-only、NDEBUG 空展开（**天然依赖预处理器**）；`NN_ASSERT_ALWAYS`/`NN_FATAL` 无条件终止。
  - `NN_UNWRAP/NN_EXIT` 依赖 `NN_DETAIL_PICK2/3` 参数计数分发（L97-L108）。
- `src/` 宏使用规模：约 65 个文件、~180 处（`NN_EXIT` ~60、`NN_TRY*` ~70、`NN_UNWRAP` ~15、断言 ~30、`NN_FAIL` 若干）。
- 显式 include 宏头的文件（`src/`）：`core_errors.hpp` × 7（`dataset_gen`、`tokenizer_train`、`tokenizer_infer`、`dataset_convert`、`run_driver_test`、`batch_source_test`、`error_macro_test`）；`core_assert.hpp` × 1（`assert_test`）。

### 2.3 模块与构建现状

- 现有 4 个薄接口模块：`modules/nn.{core,kvrecord,dataset,tokenizer}.cpp`，DAG `core→kvrecord→dataset→tokenizer`，薄模式 = `module;` GMF 文本包含 + `export module nn.<域>;` + `export namespace nn { using nn::X; … }`（实体挂全局模块，与 include 消费方同一实体、可混链）。
- 双轨（`#ifdef NN_USE_MODULES / import / #else / #include / #endif`）共 **8 个文件**：`src/dataset_gen.cpp:13-17`、`tokenizer_train.cpp:12-16`、`tokenizer_infer.cpp:12-16`、`dataset_convert.cpp:13-17`、`dataset_test.cpp:27-31`、`nnvocab_test.cpp:19-23`、`tokenizer_consistency_test.cpp:23-27`、`dataset_integration_test.cpp:25-30`。
- `nn_use_modules` 调用 8 处：L675-L676、L682、L688（app）、L1054-L1057（test）。**`tokenizer_consistency_test` 未见调用——实施时核对补齐**（其源文件由 `tokenizer_test.cpp` 文本 include）。
- 目标创建：`nn_add_executable` 是 function（L437-L450，自动链 `nn_core`、登记 `NN_EXECUTABLE_TARGETS` 与 `NN_PCH_CANDIDATES`）。
- PCH：`nn_pch` OBJECT 库 + `target_precompile_headers(REUSE_FROM)`（L1202-L1233，宿主 `cmake/nn_pch_stub.cpp`）；模块消费方在 `nn_use_modules` L608 被剔除候选（理由 L594-L597）。
- `src/` include 面：107 个 `.cpp` 中 56 个含 `#include <neuralnet.cpp/...>`（142 处），其中 35 个直接 include `nn.hpp`（nn.hpp 共 46 行 include，见附录 B）；本地头仅 `src/expr_fuse_shapes.hpp`（:14 include `nn.hpp`）。
- `nn.hpp` 之外的细粒度消费面：expr_*（compute_tensor/expr_dsl/expr_spec/expr_opt/expr_registry/expr_fold）、`cli_engine_factory`、`kernel_cache`、`core_{sha256,file,threadpool,parallel}`、`model_keyvalue_record`、`batch_source`、`run_driver`、`backend/compute_vk_backend`、`test_common.hpp` 等（完整清单以第 5.2 节枚举命令为准）。
- `NN_GPU_ENGINE` 在 `src/` 仅作 `#ifdef` 条件（13 文件），**无本地 `#define`**（CMake 传入）；`NN_HAS_VULKAN` 仅 `fused_gpu_test.cpp:34`。
- STL：域头基本自包含（`domain_tokenizer_charbpe.hpp` 含 24 个 STL 头），但历史上有传递依赖缺口（`tokenizer_infer` 缺 `<memory>`，`history.md:5522`）——import 后消费者必须自带，靠编译错误驱动补。

### 2.4 性能基线（`research/modules-import-nn/README.md` §6.3，L107-L114）

| 场景 | include+ccache | 4 模块 + PCH |
|---|---|---|
| 改 consumer 源全量 | 0.48s | 0.50s |
| 改公共头全量增量 | 12.7s | **18.0s**（0.10+0.36+10.6+1.5+0.4+1.2+1.3+1.4+0.5） |
| 冷构建（520→348 TU） | 87.2s | 106.7s（新增 BMI 10.6s，PCH 扫描 348→31） |

同时记住：**全量单体 `import nn`（86MB BMI）已被证伪**（§1，L16-L28：比 ccache 命中慢 19×）。本方案走 §5 认可的**粒度模块**路线，不走单体。

---

## 3. 已定决策（含备选与否决理由）

| # | 决策 | 备选与否决理由 |
|---|------|----------------|
| D1 | **BMI 命令 = 目标编译命令镜像**，公共选项提取单一真源变量 | 否决「只补 `-pthread` 最小修」：其余 5 处差异（尤其工具链 flags 缺失）是同类隐患 |
| D2 | **无编译器识别**：删 `NN_ENABLE_MODULES`/`NN_MODULES_ENABLED`，模块构建无条件；非模块编译器构建直接失败 | 用户定调。BMI 姿势保持 Clang/GNU 驱动（`--precompile`、`-fmodule-file=`）不分叉 |
| D3 | **宏岛头**：宏段拆到**不带 `neuralnet.cpp/` 前缀**的 `engine/abstraction/include/nn_error_macros.hpp`，`core_errors.hpp`/`core_assert.hpp` include 它（库内单一真源、零改动）；src/ 消费方 include 该头 + `import nn.*` | 否决「宏全函数化」：`NN_TRY*` 是 early-return（~70 处逐点改写、函数替代不了）、`NN_ASSERT` 的 NDEBUG 语义天然要预处理器；函数化无法零改写达标 |
| D4 | **AOT 收集链直接删除、不留开关**（`scan_exprs`、`fused_registry.hpp` 生成、`NN_EXPR_SCAN` 分支、`nn_enable_gpu_fusion()`、depfile/glslc 扫描步） | 与 D2「不保留开关」一致；`docs/internal/33` L125-L131 已给出可拆清单。若 P3 验证门失败，仅在那四个入口局部补救，不恢复全链 |
| D5 | **预热双入口**：① 运行期 `model_warmup()` 接线进 GPU 训练/推理入口（含 `--no-warmup` 逃生阀）；② 构建期新增 `nn_warmup` 目标（常用 layer × 代表性 shape 过 GPU forward，产物 `kernels.nnk` 落 `NN_KERNEL_CACHE_DIR`，随发布物分发） | 用户提出的两种做法都做：①消除人工遗漏，②消除冷启动 JIT 与部署环境对 kernelgen 的首跑依赖 |
| D6 | **`NN_STRICT_CLOSED_WORLD` 语义重定义**为「只用预生成缓存、miss 即报错」；默认 OFF 不变。实现时若代码形态不允许则删除该开关。**裁定：该开关不存在，不保留旧行为兼容** | 闭合世界随 AOT 注册表一起失去对象，保留旧语义等于不可用 |
| D7 | **粒度模块全量铺开**（nn.hpp 46 头 → ~8 个新域模块），export 清单先试 `export using namespace nn;`，不接受再逐符号 | 否决单体（已证伪，§2.4）；`export { #include }` 包裹已被 §4.1（L54-L55）证死（STL 私有头撞全局片段防护） |
| D8 | **`nn_add_executable` 自动 `nn_use_modules(<全部域>)`** 单点接线 | 否决 107 处逐目标登记：错误成本高、无信息增量 |
| D9 | PCH 机制随 src/ 全出候选后**评估整体删除**（`nn_pch`、`cmake/nn_pch_stub.cpp`、L1202-L1233 块），候选为空时至少保持自动空转 | 保留给范围外目标的可能性在实施时看剩余候选数决定 |

---

## 4. 实施前置

1. `git submodule update --init --depth 1 engine`（**当前未检出**，`git submodule status` 前缀 `-d2dd5d1…`；如 https 被拒：`git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init engine`——本沙箱已验证可行）。
2. 工具链：CMake **≥3.30**（沙箱自带 3.28.3 不满足 `cmake_minimum_required(3.30)` → `pip install cmake`）；clang **≥18**（沙箱默认 clang 17 不识别 `-std=c++26`，`clang-18` 已可用）；ninja。
3. 基线留存：改造前跑一遍 `cmake --preset dev && cmake --build --preset dev` + `ctest`，留存全绿基线；`ninja -t commands | grep nn.tokenizer.cpp` 留存 BMI 命令样本供 P0 diff。
4. 一切改动**按阶段提交**，每阶段独立可回滚、验收通过才进入下一阶段。

---

## 5. 分阶段任务

### P0 · 编译选项统一（治 config mismatch）

**目标**：第 0.2 节差异表归零；模块消费方构建零报错。

改动点（全部在根 `CMakeLists.txt`）：

1. 提取 `NN_COMMON_COMPILE_OPTIONS` 单一真源变量，承载：`-fno-exceptions -Wall -Wextra -Wpedantic -Werror`（L128）、GNU 分支 `-Wno-maybe-uninitialized`（L134）、Clang 分支 `-fexperimental-library -Wno-pass-failed`（L160）、LTO 开启时 `-flto=thin`（L145）。`add_compile_options(${NN_COMMON_COMPILE_OPTIONS} ...)` 与 BMI `_flags` **共用同一变量**。
2. `nn_add_module`（L552-L562）`_flags` 改为镜像组装，顺序建议：
   - `separate_arguments(_user UNIX_COMMAND "${CMAKE_CXX_FLAGS}")` → 调用方/工具链 flags（差异 #2）；
   - `CMAKE_BUILD_TYPE` 非空时再拆 `${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}`（差异 #3，含 `-DNDEBUG`/`-g`；重复的 `-O3` 无害）；
   - `${NN_COMMON_COMPILE_OPTIONS}`（含 `-flto=thin`，若 `--precompile -flto=thin` 在实测中不接受则排除并在注释说明「LTO 不进入 PCM 配置检查」）；
   - `${_opt}`（现有 DEBUG/RELEASE_FLAGS，保持原逻辑）+ `-std=c++${CMAKE_CXX_STANDARD}`；
   - **`-pthread`**：`if(CMAKE_USE_PTHREADS_INIT)`（`find_package(Threads)` 在 L256、早于模块注册 L615，configure 期可判；与 `Threads::Threads` 展开值严格一致）（差异 #1）；
   - `-DNN_ENABLE_PROBES`（已有）+ `$<$<TARGET_EXISTS:nn_gpu>:-DNN_GPU_ENGINE=1>`（生成期求值，规避 L615 早于 L1178 的顺序问题）（差异 #4）；
   - `-I` 补齐为与 L345-L350 `nn_core` 完全相同的 5 个（差异 #5）；
   - 不加 `NN_USE_MODULES`（D2/0.2 注）。
3. 清理重建：`rm -rf build/<preset>/modules/*.pcm`（或整个 build 树）。
4. 验证（**验收门 A**）：
   - `tokenizer_infer / tokenizer_train / dataset_gen / dataset_convert / dataset_test / tokenizer_test / nnvocab_test / dataset_integration_test` 全部构建通过，config mismatch 与级联误报归零；
   - `ninja -t commands` 取 BMI 命令与 `tokenizer_infer` 编译命令 diff，**允许差异仅** `-fmodule-file` / `-o` / `--precompile` / `-c` / 输出路径；
   - 全量 dev preset 构建 + ctest 与基线一致。

### P1 · 拆开关：无条件化、无编译器识别、删双轨

1. 删 L471-L482：`option(NN_ENABLE_MODULES ...)` 与 `if(... AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT MSVC)`；`NN_MODULES_ENABLED` 条件全部移除（`nn_add_module` 开头 L539、`nn_use_modules` L599 等处的 `if(NOT NN_MODULES_ENABLED)` 分支删净）。模块构建无条件执行。
2. 删 `NN_USE_MODULES` 宏传播：`nn_mod_*` INTERFACE 的 `target_compile_definitions(... NN_USE_MODULES)`（L579-L585 内）。
3. 删 8 处双轨（文件清单见 2.3）：`#ifdef NN_USE_MODULES … #else #include … #endif` → 无条件 `import nn.<域>;`（域按各自现有 import 行）。
4. 注释/文案同步：L466-L469「仅 GNU 驱动 Clang…自动 OFF」、option help 中 `(clang only; ...)`（随 option 删除）、L594-L597 注释措辞。
5. 核对 `tokenizer_consistency_test` 的 `nn_use_modules` 缺漏并补齐（2.3）。
6. 验证（**验收门 B**）：全量构建 + ctest 绿；`grep -rn "NN_ENABLE_MODULES\|NN_USE_MODULES" CMakeLists.txt src/` 仅剩历史文档引用；全仓 grep 确认无编译器 ID 分支新增。

### P2 · 宏岛头

1. 在 `engine/abstraction/include/` 下**新建 `nn_error_macros.hpp`**（**不在** `neuralnet.cpp/` 子树）：从 `core_errors.hpp` 迁出宏段（`NN_TRY`/`NN_TRY_MSG`/`NN_TRY_CHECK`/`NN_UNWRAP`/`NN_EXIT`/`NN_FAIL`/`NN_DETAIL_*`），从 `core_assert.hpp` 迁出（`NN_ASSERT`/`NN_ASSERT_ALWAYS`/`NN_FATAL` 等），视 `src/` 实际使用补 `NN_VECTORIZE_PRAGMA` 等。
2. `core_errors.hpp`/`core_assert.hpp` 改为 `#include "nn_error_macros.hpp"` + 保留原声明 → **库内宏单一真源、模块 GMF 与所有头用法零改动**。
3. `src/` 侧：7 个 include `core_errors.hpp` 的文件、`assert_test.cpp`、以及所有使用宏但宏将不可见的文件（~65 个，枚举命令见附录 A）显式补 `#include <nn_error_macros.hpp>`（include 路径经 `nn_core` 的 `engine/abstraction/include` 已覆盖，无需加 `-I`）。
4. 验证（**验收门 C**）：
   - `grep -rn "^#include <neuralnet.cpp/" src/ | wc -l` = **0**（此为总门禁，P2 后应已满足——若 P5 未完成则此门在总验收执行）；实际上 P2 阶段先保证宏可见：全量构建无 `NN_* 未声明` 报错；
   - `error_macro_test`、`assert_test`、`assert_always_test` 语义回归通过（宏仍被测试覆盖）；
   - 库侧零行为变化：`modules/` 与范围外目标照常构建。

### P3 · AOT 收集 → JIT 预热

**删除**（D4）：
1. `scan_exprs` 相关 CMake（L427-L430 注释块及其目标、`fused_registry.hpp` 生成命令/depfile、glslc 扫描步）；
2. `nn_enable_gpu_fusion(<target> MAIN <src>)` CMake 函数（`docs/usage/05` 引用它 → 文档同步）；
3. 头文件内 `#ifdef NN_EXPR_SCAN` 分支（`expr_dsl`/`expr_spec`/引擎 `end_expr` 登记钩子，位于 submodule，实施时 grep 定位逐一删）；
4. `src/expr_fuse_shapes.hpp:7` 注释与 `NN_TEST_COUNTER` 无关、与 dry-run 相关逻辑（实施时按其实际内容处置，保持 import 化一致）。

**接入预热**（D5）：
5. 运行期：在 GPU 训练/推理入口（`text_train`、`mnist_train`、`text_infer`、`mnist_infer`，及确认有 GPU engine 创建的其他 app）engine 创建后调用 `model_warmup()`；失败经 `Result` 返回，调用方 `NN_TRY_MSG(..., "预热失败", ...)` 上抛（`9672023`）；CLI 加 `--no-warmup` 逃生阀（cli 头按现有 flag 模式新增）。
6. 构建期：新增 `nn_warmup` 可执行目标（`nn_add_executable` + `nn_use_modules`），内容：常用 layer × 代表性 shape 序列过一遍 GPU forward，写 `kernels.nnk` 至 `NN_KERNEL_CACHE_DIR`（输出到 build 目录并在发布脚本中随产物分发）。它同时充当入口②的“编译目标预热”（用户方案 A），并为 `fold/reduce/into/multi` 提供显式覆盖面（见下）。
7. ~~`NN_STRICT_CLOSED_WORLD`（D6）~~：**已删除**（2026-11-08，用户裁定不保留旧行为兼容）；运行期 JIT 为唯一 miss 路径。

**验证（验收门 D——本阶段关键，决定方案成立性）**：
- 实测空注册表行为：构建临时验证（或直接删注册表后）跑 `expr_gpu_test`、`expr_fold`、`expr_reduce`、`expr_fuse_test`、`fused_gpu_test`，确认 `fold/reduce/into/multi` 四入口是否硬报错；
- 若**不**硬报错 → 预热路径直接成立；
- 若硬报错 → 首选对策：确保 `nn_warmup` 与各 app 预热**显式覆盖四类入口**（零库改动）；仍不闭合则给四入口补 JIT 兜底（改引擎，工作量单列）；
- 性能锁：`fused_gpu_test --hash` 位级对拍（预热后与 CPU 结果一致）；冷缓存首跑 vs `kernels.nnk` 命中跑耗时对比（证明秒命中）；`kernel_cache_test`、`jit_demo` 通过。

### P4 · 新增域模块（铺满 nn.hpp 46 头）

1. 域划分草案（实施中按头闭包微调，闭包按 L484-L486 铁律宁多勿少）：

| 新模块 | 覆盖头族（nn.hpp 行号见附录 B） |
|---|---|
| `nn.expr` | expr_dsl/spec/opt/graph/registry/fold、compute_tensor、tensor_schema、named_tensor_set |
| `nn.engine` | compute_engine、backend(compute_vk_*)、algebra_*、precision、ternary/int8、core_profile、kernel_cache |
| `nn.layer` | compute_layer_* 全家族 + layer_name/能力声明头 |
| `nn.loss` | compute_loss |
| `nn.opt` | compute_optimizer、lr_scheduler |
| `nn.model` | model_container/serialization/spec、train_state、training_session、training_group_session |
| `nn.train` | text_trainer、batch_source、run_driver、dataset_samples、rl_experience/policy/pipeline |
| `nn.domain` | domain_mnist/gpt/rla/cnn |
| `nn.cli` | cli/*（含 `cli_engine_factory`，其 `#ifdef NN_GPU_ENGINE` 要求 P0 的 BMI `-D` 对齐） |

既有 `nn.core` 补入未覆盖的核心头（precision 已含则不重复）。所有域 `export import nn.core`，防环。

2. 每个模块单元按现有薄模式（`modules/nn.*.cpp` 样例）：GMF `#include` 域头 → `export module` → re-export。
3. **export 清单生成法**（D7）：先写一个模块试 `export using namespace nn;`——若 clang 接受（importer 侧可见全局模块实体），全部模块降为行级文件；不接受则逐符号 `using`，以编译报错 `use of undeclared identifier` 驱动补齐（机械、量大，为本方案最大确定性工作量）。
4. CMake：为每个新域写 `NN_MOD_*_HEADERS` 闭包 + `nn_add_module(...)` 注册（放 L615-L625 现有注册旁，保持依赖序）；DAG 正确性靠 `-fmodule-file` 依赖链（`nn_add_module` 的 DEPS 参数）。
5. 验证（验收门 E）：各 BMI 独立构建成功；`ninja -t commands` 复查 BMI 与消费方命令 diff 仍仅模块差异；`tokenizer_infer` 等既有消费方零改动仍绿。

### P5 · src/ 全量切 import + CMake 自动接线

1. `nn_add_executable`（L437-L450）内部自动调用 `nn_use_modules(<全部域>)`（D8）——自动挂全量 `-fmodule-file`、自动移出 PCH 候选（L608 逻辑复用）。
2. 源码改造（枚举见附录 A，逐文件核对）：
   - 35 个 include `nn.hpp` 的文件 → 拆成所需域 `import nn.<域>;`（按实际符号使用，宁多勿少）；
   - 细粒度 include 文件（expr_*、cli、core_*、kernel_cache、backend 等）→ 对应域 import；
   - 本地头 `src/expr_fuse_shapes.hpp` 同步 import 化；
   - 宏岛 include 按 P2 清单补全；
   - `test_common.hpp` 保持文本 include。
3. STL 可见性：import 后由编译错误驱动逐文件补 `#include <memory>` 等（消费者自带，惯例与既有 9 消费方一致）。
4. PCH（D9）：src/ 全出候选后检查 `NN_PCH_CANDIDATES` 剩余量；为空则删除 `nn_pch` 机制与 `cmake/nn_pch_stub.cpp`（L1202-L1233、L1220-L1224），否则保留注释说明服务对象。
5. 验证（验收门 F）：
   - `grep -rn "^#include <neuralnet.cpp/" src/ | wc -l` = **0**（总门禁）；
   - dev preset 全量构建 + ctest 全绿（与基线用例集一致）；
   - 每目标编译命令含其 import 所需 `-fmodule-file`；
   - 范围外目标（tools/examples/downstream）不受影响。

### P6 · 收尾：性能对照、门禁、文档同步

1. **编译增量对照**（按 §2.4 口径）：改 consumer 源全量、改公共头全量增量、冷构建 三场景，与基线对比记录；实测 import TU 是否还能被 ccache 命中（预期：不能）。
2. **运行性能**：预热命中 vs 冷缓存首跑；与改造前 GPU 基线（`benchmarks/` 既有脚本）对拍关键用例。
3. **门禁固化**：把 `grep "^#include <neuralnet.cpp/" src/` = 0 做成 configure 期检查（`execute_process` + FATAL_ERROR）或 CI 步骤；BMI/消费方命令 diff 可留作一次性人工核验记录。
4. **文档同步**（改现有文档，不新建）：
   - `AGENTS.md`：能力表（「仅 Clang」→ 无编译器门；双轨消费者 → 一律 import；`NN_ENABLE_MODULES`/`NN_ENABLE_PCH` 开关清单）、铁律段（NN_EXPR_SCAN 条目按 D4 改写为「已由 JIT 预热替代」）；
   - `research/modules-import-nn/README.md`：§6 状态更新（双轨已消灭、开关已删、收集器已拆）；
   - `docs/development/01-cmake-integration.md` §8.1：模块接入说明随 D2/D8 更新；
   - `docs/usage/05-consume-as-library.md`：`nn_enable_gpu_fusion` 删除说明；
   - `docs/development/04` FAQ：strict 语义（D6）与 JIT 预热；`docs/development/02` 宏章节：宏岛头路径；
   - `docs/history/history.md`：本次迁移记录。

---

## 6. 验收总表

| 门 | 阶段 | 判据 |
|----|------|------|
| A | P0 | 模块消费方零 config-mismatch；BMI/消费方命令 diff 仅模块相关差异；全量构建 ctest 与基线一致 |
| B | P1 | 无 `NN_ENABLE_MODULES`/`NN_USE_MODULES`/编译器 ID 判断；8 文件无条件 import；全绿 |
| C | P2 | 宏单一真源；`error_macro_test`/`assert_test` 绿；库侧行为零变化 |
| D | P3 | 空注册表下四入口行为实测结论 + 对策落地；`--hash` 对拍绿；冷/热耗时数据 |
| E | P4 | 全部新 BMI 构建成功；命令 diff 复查；既有消费方零改动仍绿 |
| F | P5 | `src/` 下 `<neuralnet.cpp/` include = 0；全量 ctest 绿；范围外目标不受影响 |
| G | P6 | 增量/运行性能对照记录；门禁固化；9 处文档同步完成 |

---

## 7. 风险登记册

| 级别 | 风险 | 对策（含触发后的回退） |
|------|------|------------------------|
| 高 | `fold/reduce/into/multi` 空注册表硬报错，预热无法覆盖 | 验收门 D 实测；预热显式覆盖四入口（零库改动）；仍失败则给四入口补 JIT 兜底；**最末手段**：仅为这四类保留窄收集器（违反 D4，需回评审） |
| 高 | `export using namespace nn;` 不被 clang 接受 → 逐符号清单量大 | 错误驱动脚本化补齐；域拆细以缩小单模块失败面 |
| 中 | import TU 失去 ccache 命中 → 增量劣化 | 验收门 G 实测；BMI 快路径（0.50 vs 0.48s 基线）可接受；不可接受则评估 `ccache` 模块模式或接受现状 |
| 中 | 冷缓存部署依赖 `libnn-kernelgen.so` + `glslc` | `nn_warmup` 产物随发布分发；文档标注运行时依赖 |
| 中 | 拆宏岛头改动 submodule（engine 仓）→ 跨仓发版节奏 | submodule 指针随本次提交锁定；`core_errors.hpp` 仅做拆分不改语义 |
| 中 | STL 可见性缺口逐文件暴露 | P5 逐文件编译驱动补齐；保留 history:5522 教训提示 |
| 低 | `-flto=thin` 与 `--precompile` 兼容性 | P0 实测；不接受则排除并注释（不影响 PCM 配置检查） |
| 低 | `tokenizer_consistency_test` 接线缺漏 | P1 第 5 步核对 |
| 低 | 文档漂移 | 验收门 G 九处清单逐项勾销 |

---

## 附录 A · 枚举与验证命令

```bash
# src/ 残留 include（总门禁，目标 0）
grep -rn "^#include <neuralnet.cpp/" src/ | wc -l
# 双轨残留
grep -rn "ifdef NN_USE_MODULES" src/ CMakeLists.txt
# 宏可见性改造面（P2 第 3 步）
grep -rln "NN_TRY\|NN_EXIT\|NN_UNWRAP\|NN_ASSERT\|NN_FAIL" src/*.cpp src/*.hpp
# include 面（P5 第 2 步）
grep -rn "#include <neuralnet.cpp/" src/
# BMI 命令样本 / 消费方命令（P0 diff）
ninja -C build/<preset> -t commands | grep -A2 "nn.tokenizer.cpp"
ninja -C build/<preset> -t commands | grep "tokenizer_infer.cpp"
# 开关/编译器识别残留
grep -rn "NN_ENABLE_MODULES\|NN_MODULES_ENABLED\|NN_USE_MODULES\|CMAKE_CXX_COMPILER_ID" CMakeLists.txt
```

P0 BMI 与消费方 diff 的**允许差异集**：`-fmodule-file`、`-o`、`--precompile`、`-c`、输出路径。

## 附录 B · nn.hpp 的 46 行 include（域划分输入）

`include/neuralnet.cpp/nn.hpp` L16-L61：

- 形式头（`<neuralnet.cpp/...>`）：core_errors(16)、core_assert(17)、core_threadpool(19)、core_file(20)、core_config(21)、dataset(52)、domain_tokenizer(61)
- 相对 include（`"..."`）：core_observer_ptr(18)、algebra_matrix(24)、compute_tensor(31)、tensor_schema(32)、named_tensor_set(33)、expr_spec(34)、compute_engine(35)、expr_dsl(36)、compute_layer(38)、compute_loss(39)、compute_optimizer(40)、rl_experience(41)、rl_policy(42)、rl_pipeline(43)、model_container(46)、training_session(47)、training_group_session(48)、train_state(49)、model_spec(50)、model_serialization(51)、dataset_samples(53)、lr_scheduler(54)、domain_mnist(57)、domain_gpt(58)、domain_rla(59)、domain_cnn(60)
- 不进聚合头：`text_trainer.hpp`（L63-L65 注）——import 面按文件实际使用补齐

## 附录 C · P3 关键证据锚点

- `include/neuralnet.cpp/model_container.hpp:350-362` —— `model_warmup()` 定义（待接线）
- `docs/internal/33-kernel-cache-and-engine-decouple.md:39-59,125-131` —— JIT 架构与「可拆 scan_exprs」设计
- `src/fused_gpu_test.cpp:563-603` —— JIT 路径正确性回归
- `docs/development/04-compute-engine-development.md:863-876` —— 四入口硬报错与 strict 语义（待代码复核）
- `AGENTS.md:483-485` —— GPU 路径能力总览
- `docs/release-notes/v1.6.0.md:60,91,103`、`v1.6.1.md:16,49-50,68`、`v1.7.0.md:129` —— `NN_EXPR_SCAN` 相关历史不变量（P3 删除时逐条核对是否失效）
