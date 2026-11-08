# 构建集成：把库接进你的工程

> **读者**：调用者（库使用者）——要在自己的 CMake 工程里用 neuralnet.cpp，或要在本仓
> 加文件 / 加 shader / 加测试。总索引见 [`README.md`](README.md)。
>
> 消费方的最小接入也可以看 [`../usage/05-consume-as-library.md`](../usage/05-consume-as-library.md)；
> 本文是**完整版**：全部构建开关、GPU 语义、自研算子融合注册、跨平台与故障排查。
> **状态**：现行。

---

## 1. 两种角色

| 角色 | 你做什么 | 用哪套 CMake |
|---|---|---|
| **消费方**（把库嵌进自己的工程） | `add_subdirectory` + 链 `neuralnet::nn` | 本文 §2–§6 |
| **本仓开发**（改库本身 / 跑测试） | 在仓库内构建 app / 测试 | 本文 §7 |

目标就一个：**门面目标 `neuralnet::nn`**。把 C++ 标准、include 目录、线程库、GPU 宏与
shader 生成顺序全部打包传递，你只写两行。

---

## 2. 最小接入（两行 CMake + 一个 include）

```cmake
cmake_minimum_required(VERSION 3.30)
project(my_app LANGUAGES CXX)

# ① 嵌入库（把仓库放到源码树任意位置；下例为 3rd_party/）
add_subdirectory(3rd_party/neuralnet.cpp)

# ② 链接门面目标
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

```cpp
// main.cpp
#include <neuralnet.cpp/nn.hpp>

int main()
{
    auto engine_owner = nn::CpuEngine::create();
    nn::CpuEngine& engine = *engine_owner;
    nn::Model model = NN_UNWRAP(nn::build_mnist_mlp_model(engine));
    (void)model;
    return 0;
}
```

```bash
cmake -S . -B build -G Ninja && cmake --build build
```

**FetchContent** 同理，把 ① 换成：

```cmake
include(FetchContent)
FetchContent_Declare(neuralnet.cpp GIT_REPOSITORY <url> GIT_TAG <tag>)
FetchContent_MakeAvailable(neuralnet.cpp)
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

独立最小样例工程见 [`examples/downstream/`](../../examples/downstream/)（CPU / GPU 双路径 + ctest 注册）。

---

## 3. `neuralnet::nn` 替你传递了什么

链接它之后，你的目标自动获得：

- **C++ 标准**：`cxx_std_26`（Clang/GNU）；MSVC `cl` 不认识 `cxx_std_26` →
  自动降级 `cxx_std_23`（映射 `/std:c++latest`）；
- **include 目录**：`include/` +（GPU 时）SPIR-V 嵌入头目录 `generated/`；
- **线程库**：`Threads::Threads`；
- **GPU（若启用）**：宏 `NN_HAS_VULKAN`、`Vulkan::Vulkan` 链接、以及
  「shader 生成先于你的 TU 编译」的顺序依赖；
- **库头需要的接口选项**：Clang 的 `-Wno-pass-failed` / `-fexperimental-library`、
  MSVC 的 `/utf-8`、`/Zc:preprocessor`（否则 `NN_UNWRAP` / `NN_EXIT` 会被
  传统预处理器静默展开错）。

> 你**不要**自己写任何 Vulkan 相关 CMake。源码里 `#ifdef NN_HAS_VULKAN` 分支即可。
> ⚠ `Vulkan::Vulkan` 是 imported 目标，**不可跨目录直接引用**——一律经 `neuralnet::nn` 传递。

---

## 4. 构建开关（在 `add_subdirectory` **之前**设置）

| 变量 | 取值 | 默认 | 含义 |
|---|---|---|---|
| `NN_ENABLE_GPU` | `AUTO` / `ON` / `OFF` | `AUTO` | `AUTO` = 探测到 Vulkan + glslc 就启用 GPU；`ON` = 强制要求，缺依赖则**配置期失败**（不静默降级）；`OFF` = 完全跳过（不探测、不生成 shader → 纯 CPU 最快路径） |
| `NN_BUILD_APPS` | `ON` / `OFF` | **嵌入时 OFF**（顶层工程 ON） | 是否构建本仓 app / 基准可执行文件。嵌入时默认 OFF —— 消费方只拿库目标 |
| `NN_ENABLE_TESTS` | `ON` / `OFF` | `OFF` | 是否构建并注册本仓测试目标（`list(APPEND NN_TEST_TARGETS ...)` 收集） |
| `NN_ENABLE_NATIVE` | `ON` / `OFF` | `ON` | `-march=native`（只作用于本仓目标，不影响消费方编译选项）。分发 / CI 用 `OFF` 生成可移植基线 |
| `NN_ENABLE_PROBES` | `ON` / `OFF` | `OFF` | **探针编译门**：打开才编译库内热路径探针并创建本仓探针目标（`f16_cpu_probe` / `gpu_stability_probe`；GPU 显存分项探针 `mem_probe` 已随引擎 .so 化迁往新仓库 `compute.neuralnet.cpp`）。会改变库头语义 → **不可与默认构建的对象 / 缓存混用** |
| `NN_COMPILER_LAUNCHER` | path | 自动探测 ccache / sccache | 编译器缓存启动器；显式传空串 = 禁用。**模块消费方一律绕开**（ccache 4.13 不支持标准 C++ modules，见「C++ 模块」小节） |

```cmake
# 例：纯 CPU、不编本仓应用
set(NN_ENABLE_GPU OFF)
add_subdirectory(3rd_party/neuralnet.cpp)
```

命令行等价：`cmake -B build -DNN_ENABLE_GPU=OFF`。

> **探针门语义**：`NN_ENABLE_PROBES` 经 `nn_core` 的 **INTERFACE** 定义传播
> （`NN_ENABLE_PROBES=1`），保证所有 TU 取值一致（同头两种语义 = ODR 隐患）。
> 关时 `nn_dbg_scan` / `NN_PREC_TRACE` / `NN_NAN_TRACE` / `NN_MEM_STATS` 等**不参与编译**，
> 相关环境变量一律无效。默认关 → 正常训练/推理没有额外分支与字符串。

---

## 5. GPU 语义

- **能力随目标传递**：链接 `neuralnet::nn` 且配置期 GPU 启用，你的目标就自动拿到
  `NN_HAS_VULKAN`、SPIR-V 嵌入头目录、`Vulkan::Vulkan`、shader 生成顺序。
- **`NN_ENABLE_GPU=AUTO` 的静默降级**：找不到 Vulkan / glslc 时走纯 CPU。
  想要「没 GPU 就配置失败」用 `ON`。
- **表达式执行**：库内层的结构在 GPU 首次执行时按 `expr_spec_key` 走运行期 JIT
  （懒加载 `libnn-kernelgen` → SPIR-V → 写 `.nnk` 缓存），预热可用 `model_warmup()`。
  你在自己的层里写的 `nn::dsl::compute` 表达式**无需任何构建期步骤**——同一套 JIT 路径覆盖，
  首次有一次性编译延迟（之后命中缓存）。

---

## 6. 自研算子的 GPU 执行：运行期 JIT（无需 CMake 步骤）

构建期 AOT 收集链（`scan_exprs` / `fused_registry.hpp` 生成 / `NN_EXPR_SCAN` 分支 /
`nn_enable_gpu_fusion()`）已随 P3/D4 整体删除，**本仓与 engine 子仓都不再有这个函数**。

自研 `nn::dsl::compute` 表达式在 GPU 上的执行路径与库内层完全一致：

1. **首次 miss**：引擎按 `expr_spec_key` 未命中 `.nnk` 缓存 → 懒加载 `libnn-kernelgen`
   （`NN_KERNELGEN_PATH` 可覆盖路径）→ 现编 SPIR-V → 建 pipeline → 执行；
2. **落盘缓存**：生成的 kernel 写入 `NN_KERNEL_CACHE_PATH` / `NN_KERNEL_CACHE_DIR` 指定的
   `.nnk`（默认用户缓存目录），后续进程直接命中；
3. **预热**：`nn::model_warmup()` 在加载模型后跑一遍前向，把冷启动成本提前到部署首跑；
   也可用构建期/发布期预热目标（见 CMakeLists 的 `nn_warmup` 相关目标与 `--no-warmup` 语义）。

```cmake
add_executable(my_app main.cpp my_layer.cpp)          # 自研层在 my_layer.cpp
target_link_libraries(my_app PRIVATE neuralnet::nn)
```

无需额外 CMake 行；本仓样例 `examples/downstream/` 演示最小消费形态。

---

## 7. 本仓开发：构建与测试

```bash
# 推荐：用仓库自带的 CMakePresets.json（cmake --list-presets 看全部）
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
cmake --workflow --preset dev          # 配置 + 构建 + 测试 一条命令

# 等价的手写写法：
# 默认 Release：-O3 -funroll-loops -march=native
cmake -B build -G Ninja && cmake --build build

# 开测试（测试程序在 build/，注册进 ctest）
cmake -B build -G Ninja -DNN_ENABLE_TESTS=ON && cmake --build build && ctest --test-dir build
```

预设一览（`binaryDir` = `build/<preset名>`）：

| preset | 配置 | 用途 |
|--------|------|------|
| `release` | Release + `-march=native`，无测试 | 跑应用，最省时间 |
| `dev` | `release` + `NN_ENABLE_TESTS=ON` | 本仓日常开发（默认选它） |
| `debug` | `dev` 换 `-O0 -g` | 断点/单步 |
| `portable` | `dev` + `NN_ENABLE_NATIVE=OFF` | 发布/CI/换机可跑的基线 |
| `cpu` | `dev` + `NN_ENABLE_GPU=OFF` | 纯 CPU 最短构建路径（不探测 Vulkan、不生成 shader） |
| `probes` | `dev` + `NN_ENABLE_PROBES=ON` | 排障探针；⚠ 库头语义不同，缓存/二进制不可与正常构建混用 |

构建出的应用入口：`build/{mnist_train,mnist_infer,text_train,text_infer,tokenizer_train,
tokenizer_infer,dataset_gen,dataset_convert}`，另有 `layer_bench`（性能）与测试类可执行文件。
`NN_BUILD_APPS` 控制这批 app；探针目标（`f16_cpu_probe` / `gpu_stability_probe`；`mem_probe` 已随引擎 .so 化迁往新仓库）
另受 `NN_ENABLE_PROBES` 控制。

**已知构建坑**：链接报 `permission denied`（`build/` 下 exe 正在运行被占用 → 构建中止）
→ 追加 `-- -k 0` 让 ninja 跳过失败目标继续建其余目标（被占用目标仍需关掉进程后重链）。

### IDE（clangd）如何解析 `<neuralnet.cpp/xxx.hpp>`

include 路径全部来自 `target_include_directories`（`include/` +
`build/generated`），clangd 必须拿到编译数据库才能解析。顶层配置时 CMake 自动
导出 `build/compile_commands.json`（`CMAKE_EXPORT_COMPILE_COMMANDS`，仅 Ninja/Makefile
生成器有效）；仓库根的 `.clangd` 已指向 `build/`，clangd 打开任意头/源文件即可。
用 preset 构建时编译数据库在 `build/<preset名>/`，把 `.clangd` 的
`CompilationDatabase` 改成对应目录（或传 `--compile-commands-dir`）。
改完 CMake 后记得重新 configure 一次，让数据库里的 `-I` 与目标保持同步。

---

## 8. 编译耗时

本工程 header-only，单 TU 首次 parse `nn.hpp`（含 44 份原语 SPIR-V 嵌入头 + AOT 融合注册表，
约 7MB 头文本）较慢——本机实测约 17.5s 无 PCH。src/ 全量模块化（P5，docs/internal/35 D8）
后**共享 PCH 机制已整体删除**（D9），消费方经 `-fmodule-file` 从 BMI 取声明；落地提速项
（默认开启，无需配置）：

1. `CMAKE_CXX_SCAN_FOR_MODULES OFF`：本仓 C++ 模块走**显式接线**（见 8.1「C++ 模块」小节），
   不开 CMake 模块扫描 → 只有 13 个模块单元参与，其余 ~270 个 TU 零扫描；
2. `scan_exprs` 的 glslc **并行**（环境变量 `NN_SCAN_JOBS=<n>` 覆盖，`=1` 回串行）
   + `fused_registry.hpp` 内容稳定写入（内容不变则不动 mtime，靠 `restat` 掐断下游重编）；
3. 编译器缓存探测扩到 ccache / sccache（可用 `-DNN_COMPILER_LAUNCHER=<path>` 指定），
   但**模块消费方一律绕开 launcher**（ccache 4.13 不支持标准 C++ modules，key 不含
   `-fmodule-file` 的 BMI 内容 → "BMI 重编 + 消费方源码没动"时会吐出旧 .o）；
4. **BMI 是消费方 TU 的真实编译依赖**（P6 修复，2026-11-06）：`nn_use_modules` 经
   `OBJECT_DEPENDS` 把 pcm 挂进每个源文件编译边——`add_dependencies` 在 ninja 里是
   order-only（pcm mtime 变不触发 TU 重编）且 clang 的 `-MD` depfile 不记录 pcm，
   缺了它改公共头只重编 BMI、消费方 .o 保留旧实例化（陈旧代码）。

**实测数字**（本机 8 核，clang 24 / Linux / Release + 测试 + GPU，P5/P6 后口径）：

| 口径 | 时间 |
|------|------|
| 冷构建（无缓存） | 见 `research/modules-import-nn/README.md` §6.3（PCH 时代 13m26s / 189 边；P5 后每 TU 全量解析，更慢） |
| 改 consumer 源（mnist_train 单 TU + LTO 链接） | **≈79 s**（无 ccache 命中、无 PCH 的解析+优化+代码生成） |
| 改公共头 → 全量增量 | **≈35 min**（82 消费方 TU + 13 BMI + 全链接；`OBJECT_DEPENDS` 传导，正确性必需） |

> 历史归因（PCH 时代，`-ftime-trace`）：后端 29.1 s / 总计 33.1 s（优化器 16.6 s、代码
> 生成 12.5 s），前端只占 4.0 s（无 PCH 时前端 16.8 s、总计 45.8 s）——**编译时间大头是
> LLVM 优化/代码生成**，不是头解析。此结论 P5 后仍然成立（PCH 删后前端回归全量解析，
> 每 TU 成本整体上移，见 §8.1）。

---

## 8.1 C++ 模块（粒度导入，无条件构建）

自 2026-11-01 起本仓提供**粒度 C++20 模块**消费路径；P5（2026-11-06）铺满 **13 个域模块**
（`nn.core` / `nn.kvrecord` / `nn.dataset` / `nn.tokenizer` + expr/engine/layer/loss/opt/
model/train/domain/cli，DAG 叶子链 `core→kvrecord→dataset→tokenizer`、P4 新域全部 DEPS core），
src/ 全量 import 化。动机与实测结论见 `research/modules-import-nn/README.md` §6
（**非提速路径**，价值 = 标准符合性 + 精确依赖 + 摆脱 PCH）。

- **开关**：**无**——模块构建无条件（P1 删 `NN_ENABLE_MODULES`/`NN_USE_MODULES` 宏传播、
  P5/D9 删 `NN_ENABLE_PCH`）；仅 **GNU 驱动 Clang** 支持（GCC / cl.exe / clang-cl 构建直接失败，35 号 D2）。
- **形态**：每个模块单元 = 全局模块片段文本包含域头 + `export module nn.<域>` +
  `export namespace nn { using nn::X; … }` 逐符号 re-export + `export import nn.<依赖域>`
  （C++ 模块的 import **不可传递**，必须显式 re-export，消费方只 `import nn.tokenizer` 即
  连带拿到依赖域名字）。
- **构建**：显式 BMI 自定义命令（`clang++ --precompile`，`-fmodule-file` 逐依赖传入）+
  INTERFACE 目标 `nn_mod_<域>` 传播标志（`nn_add_module`）。`CMAKE_CXX_SCAN_FOR_MODULES`
  保持 `OFF`——只有模块单元参与，其余 TU 零扫描。
- **消费方接入**：`nn_add_executable` 单点接线（P5/D8）——全部可执行目标自动
  `nn_use_modules(<目标> ${NN_ALL_MODULES})`：链接模块 INTERFACE + **`OBJECT_DEPENDS` 把
  BMI 挂进每个源文件编译边**（真实依赖，见 §8 第 4 条）+ 绕开 ccache launcher。
  残留 `#include <neuralnet.cpp/…>` 由 configure 期 `nn-gate` 门禁拦截（P6-2）。
- **硬约束**：宏（`NN_TRY` / `NN_UNWRAP` 等）不可导出 → 消费方显式 include **宏岛头**
  `nn_error_macros.hpp`（P2/D3 拆出，`core_errors.hpp`/`core_assert.hpp` 经它提供宏）；
  STL 留在全局片段 → 消费方自带用到的 STL 头（缺了会报 "missing '#include'"）。`NN_EXPR_SCAN`
  宏收集链已随 P3 拆除（AOT 收集由运行期 JIT 预热替代，35 号 D4），无域再依赖该宏。
- **嵌入消费方**：`add_subdirectory` 后 `neuralnet::nn` 门面照常（`#include <nn.hpp>` 文本
  路径不依赖模块）；模块/门禁只作用于本仓 src/ 目标（`NN_BUILD_APPS` / `NN_ENABLE_TESTS`）。

---

## 9. 跨平台与编译器

- **标准要求 C++26**（接口以 `target_compile_features` 传递）。编译器不限具体版本，
  只要**支持 C++26** 即可（Clang / GCC / MSVC 均可；MSVC 走 `/std:c++latest`）。
- **CMake ≥ 3.30**。`CMakeLists.txt` 不指定编译器（由 CMake 探测或调用方
  `CXX=...` / `-DCMAKE_CXX_COMPILER` 决定），**标准库编译/链接两侧必须一致**。
- **MSVC**：必须带 `/Zc:preprocessor`（已作为接口选项随门面传递）。`cl.exe` 默认的传统
  预处理器会把 `NN_UNWRAP(x, msg)` 静默退化成逗号表达式（条件恒真、检查永不触发），
  `NN_EXIT` 直接 `C3861`。
- **Clang + libc++**：`-Werror` 下 libc++ 会对「记录类型但从未使用」的局部变量报
  `-Wunused-variable`（libstdc++ 不报）——属真实死变量，删掉即可。
- **`-fno-exceptions` 不传递**：库头不使用异常，消费方按自己的异常策略编译即可。
- 已验证四组「编译器 × 标准库」各自全新构建 + ctest：clang + libc++、
  g++ + libstdc++（CPU-only）、clang-cl + MSVC STL、MSVC cl + MSVC STL。

---

## 10. 故障排查

| 症状 | 原因 / 处理 |
|---|---|
| `target was not found: Vulkan::Vulkan` | 在自己的 CMakeLists 里直接链了 `Vulkan::Vulkan`（imported 目标不可跨目录引用）→ 改链 `neuralnet::nn` |
| GPU 报 `libnn-kernelgen` 加载失败 | JIT 生成器缺失 → 确认引擎 .so 部署目录含 `libnn-kernelgen.so`（Windows = `nn-kernelgen.dll`），或用 `NN_KERNELGEN_PATH` 指定 |
| 自研 `dsl::compute` 首次执行慢 | 运行期 JIT 冷启动 → 用 `nn::model_warmup()` 预热，或复用 `.nnk` 缓存（`NN_KERNEL_CACHE_PATH`） |
| 编译报 `cxx_std_26 not known to CXX compiler` | 编译器过旧 |
| 找不到 `neuralnet.cpp/nn.hpp` | 确认链接的是 `neuralnet::nn`，且在 `add_subdirectory` **之后** |
| 找到了 Vulkan 但想要纯 CPU | `-DNN_ENABLE_GPU=OFF`（在 add_subdirectory 之前或命令行） |
| 想「没 GPU 就失败」 | `-DNN_ENABLE_GPU=ON` |
| 探针环境变量（`NN_F16_DEBUG` 等）无效 | 默认构建不含探针 → 重新配置 `-DNN_ENABLE_PROBES=ON` |
