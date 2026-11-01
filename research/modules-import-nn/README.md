# modules-import-nn — `import nn` vs `#include <nn.hpp>` 编译速度

> **读者**：项目开发者——评估"用 C++20 模块重写库加载路径"能否提升编译速度、并符合高版本标准。
> **结论（分三层）**：① **全量 `import nn` 否决**——热路径(0.19s)比现行 ccache 命中(0.01s)慢约 19×，
> 86 MB BMI + 宏不可导出 + STL 无法干净进模块三重硬约束，实测不采纳。② **粒度导入是方向**——
> 改库加载路径的动机不止提速，还有**标准符合性**(库应是 C++20/26 模块形态而非纯 `#include`)；
> 真正的提速来自**按功能域拆窄模块**(`nn.tokenizer` 闭包仅 12 头 vs 全库 200+ 头)、消费者只
> `import` 所需域，而非全量塞进一个巨型 BMI。③ **粒度导入 2026-11-01 已落地**（四个叶子域模块
> + 9 个消费者双轨切换），但**实测其在本仓 ccache+PCH 基线下同样不是提速路径**（消费层持平、
> 头改动增量被 BMI 重建拖慢）——落地价值 = 标准符合性 + 精确依赖 + 消费方摆脱 PCH 依赖，见 §6。
> **状态**：现行（2026-11-01 实测，clang 24 / Linux / -O3 / 纯 CPU）。全量 import 不进主线；
> 粒度导入已随主线落地（NN_ENABLE_MODULES，默认 ON，仅 GNU 驱动 Clang）。

## 1. 结论

**全量 `import nn` 这条路别做。** 量化对比如下(n=3 取 min,含前后端拆分):

| 场景 | 编译耗时 | 说明 |
|------|---------|------|
| A. `#include <nn.hpp>` 冷(无缓存) | **13.58s**(前端 12.23 / 后端 1.60) | 每 TU 重 parse 全库 |
| B. module 冷 — 建 BMI | **14.85s**(一次性) | 比 A 还慢 1.3s(BMI 序列化开销) |
| B. module 冷 — 首个 consumer | 0.19s | |
| C. **include + ccache 命中(现行主线)** | **0.01s** | ← 最快路径 |
| D. module 热(改 consumer,BMI 复用) | 0.19s | 比 C 慢 **19×** |

**决定性事实**:模块热路径(0.19s)**输给** ccache 命中(0.01s)。模块能消除的是
"改一行 consumer → 重 parse 13s 全库"这件事,而 ccache 早已把它压到 0.01s 且更快。
即:**模块解决的是一个本仓不存在的问题**。

前置实测(AGENTS §7):PCH 已把单 TU 从 30.2s → 0.11s(ccache 命中);PCH 之后大头是
**LLVM 后端**(29.1s 后端 vs 4.0s 前端)。本次 A 场景的前后端拆分(前端 12.2 / 后端 1.6)
用的是最小 consumer + 无 GPU,后端占比小,但结论方向一致:**前端 parse 才是模块能碰的部分,
而它已被 ccache 归零**。

## 2. 证据

- **环境**:clang++ 24.0.0(Ubuntu)/ `-std=c++26 -O3` / 纯 CPU(无 `NN_HAS_VULKAN`,无需 `build/generated`)。
- **真实编译姿势**(逐条实测,非纸面推断;踩坑记录见 §4):
  - 建 BMI:`clang++ --precompile -o nn.pcm -x c++-module nn_module_thin.cpp`(`-c` 会退化成 ELF 对象;`-x` 必须在末尾)。
  - 消费:`-fmodule-file=nn=<pcm>`(`-fprebuilt-module-path` 对本用例不生效)。
- **BMI 可跨 consumer 复用**(实测:两个不同 consumer 复用同一 PCM 均 0 error)——这是"模块热路径"成立的前提,已验证。
- **产物**:`nn.pcm` **86.3 MB**(AST 驻留代价,见 §4)。
- **重跑**:`cd research/modules-import-nn && ./measure.sh`。

## 3. 边界(本实测**不能**说明什么)

- **不否定模块在别处的价值**:若下游是"多 TU、无 ccache、频繁改公共头"的场景,模块(0.19s/改一次)远优于重复 include(13.58s/TU)。本仓不在此场景。
- **未测 MSVC / libc++**:仅 clang + libstdc++。但结论依赖的是 ccache 命中速度(0.01s),与工具链无关,翻盘可能性低。
- **未测真实全量 TU**:A 场景用最小 consumer;真实 `text_train.cpp` 的绝对秒数更大,但那只会让"include 冷"更慢,不改变"C 最快"的排序。
- **数值/行为等价性未验证**:本 spike 只测编译速度,未对拍 `import nn` 与 `#include nn.hpp` 的运行时数值(无必要——结论已是"不做")。

## 4. 工程化硬约束(header wrapping 的真实坑,均已实测撞到)

1. **STL 无法干净进模块**:直接 `export { #include <nn.hpp> }` 会把 418 个系统头纳入 purview,
   触发 glibc/libstdc++ 私有头守卫(`bits/libc-header-start.h` 的 `#error`)。唯一可行是"薄接口"形态
   (`nn_module_thin.cpp`):文本 include 于**全局模块片段**,只 re-export 库符号。
2. **宏不随模块导出**:`NN_TRY`/`NN_UNWRAP`/`NN_ATTN_MASK_KIND_LIST` 等仍需文本 include →
   消费者被迫"import + include"双轨,接口变丑。
3. **STL 可见性缺口**:薄模块下消费者拿不到 `size_t` 等 STL 声明,须各自 `#include <cstddef>`
   (本 spike 的 `consumer_import_min.cpp` 正是如此)。
4. **86 MB BMI**:绑定编译器版本×stdlib×优化标志;ccache **不缓存 BMI 建造**(B 场景 14.85s
   每次冷构建都要重付)。四组「编译器×标准库」矩阵下 BMI 互不通用,分发成本陡增。

## 5. 落点

**本 spike 只证伪了「全量 `import nn;`」这一条路**——它比 ccache 命中慢 19×,不采纳。
但**用 C++20 模块改写库加载路径本身有独立价值**:①**标准符合性**(库停留在 header-only +
`#include`,不是 C++20/26 该有的形态;下游用模块消费、BMI 分发、`import` 语义封装都是标准能力);
②**提速**。

**真正的提速路径是「粒度导入」,不是全量导入**:不再 `import nn`(把 200+ 头全塞进一个 BMI,
86 MB、建造 14.85s),而是按功能域拆成 `nn.tokenizer` / `nn.model_serial` / `nn.engine` 等**窄模块**,
消费者只 `import nn.tokenizer`(实测其库内闭包仅 12 头,而全库 nn.hpp 是 200+ 头)。
粒度模块的 BMI 体量、建造成本、以及"改 tokenizer 不重编 engine 消费者"的隔离收益,
才是模块相对 include 的胜点;全量 import 抹掉了这些收益(所有消费者绑在一个巨型 BMI 上)。

**粒度拆分已落地（2026-11-01，见 §6）**：四个叶子域模块进主线、9 个消费者双轨切换。
落地前的两个待验项——「按功能域划模块边界 + 处理跨域符号 re-export」与「实测粒度 import vs
全量 include+ccache 的重编风暴差异」——都已兑现，实测结论见 §6.3（重编风暴差异**不占优**，
与全量 import 的结论同源：本仓 ccache+PCH 基线已把 include 重编压到 ~0.1s/目标）。

**当前主线**:`#include <nn.hpp>` + 共享 PCH(`nn_pch`)+ ccache 仍是默认加载路径;
粒度模块是**增量接入**的平行路径(9 个叶子消费者用 `import nn.<domain>`,其余消费者照旧 include),
不是一次性替换。

## 6. 粒度导入落地（2026-11-01，clang 24 / Linux / -O3 / 纯 CPU）

### 6.1 形态与边界

四个叶子域模块，DAG 依赖链 `nn.core → nn.kvrecord → nn.dataset → nn.tokenizer`，
每个都是「薄接口」模块（`modules/nn.{core,kvrecord,dataset,tokenizer}.cpp`）：

```cpp
module;                              // 全局模块片段：文本包含域头文件
#include <neuralnet.cpp/domain_tokenizer.hpp>
export module nn.tokenizer;
export import nn.core;               // 依赖域 re-export（import 不可传递，必须显式）
export import nn.kvrecord;
export import nn.dataset;
export namespace nn { using nn::Tokenizer; using nn::make_tokenizer; /* … */ }
```

- 域边界 = 合并回主仓的四个子仓域；全部不依赖 GPU/AOT 扫描。engine/layers/model/train
  域依赖 `NN_EXPR_SCAN` 宏收集（闭合世界铁律），宏不可导出 → 本轮不模块化。
- 构建 = 显式 BMI 自定义命令（`clang++ --precompile`，`-fmodule-file` 逐依赖传入）+
  INTERFACE 目标传播（`nn_mod_<域>` 带自身与传递依赖的标志 + `NN_USE_MODULES` 宏）。
  **不开 CMake 模块扫描**（`CMAKE_CXX_SCAN_FOR_MODULES OFF` 保持），~270 个非模块 TU
  零扫描成本。
- 消费方接入 = `nn_use_modules(<目标> <域...>)`：链接模块 INTERFACE + 从共享 PCH 候选
  剔除（模块消费方 TU 若同时带 nn.hpp 的 PCH 与模块 BMI，同一实体两处声明存在
  分歧风险；叶子目标编译量小，各自解析更稳）。
- 开关 = `NN_ENABLE_MODULES`（顶层默认 ON；嵌入默认 OFF；仅 GNU 驱动 Clang 生效，
  GCC/cl.exe/clang-cl 自动 OFF，消费方回落 include 分支——源码用 `#ifdef NN_USE_MODULES`
  双轨）。

### 6.2 硬约束复现（薄模块的真实代价）

| 约束 | 落地形态 |
|------|---------|
| 宏不可导出 | `NN_TRY`/`NN_UNWRAP`/`NN_ASSERT_*` 等仍文本 include（app 显式补 `core_errors.hpp`） |
| STL 可见性缺口 | 消费方自带用到的 STL 头——实测 `tokenizer_infer` 此前靠 nn.hpp 传递拿到 `<memory>`，切换后需显式补 |
| 头自包含缺口 | 模块单元按任意顺序包含域头，暴露了 `core_file.hpp` 缺 `<vector>` 的潜在 bug（正常构建靠 include 顺序侥幸通过），已修 |

### 6.3 实测（n=1，本机 / -O3 / ccache 启用 / 纯 CPU）

| 场景 | 模块 ON（import） | 模块 OFF（include+ccache+PCH） |
|------|-------------------|-------------------------------|
| BMI 体积 | core 25 MB / kvrecord 10 MB / dataset 22 MB / tokenizer 44 MB（各域独立，全量对照 86 MB） | — |
| BMI 建造（tokenizer，单线程） | 10.61 s | — |
| 改 consumer 源 → 重编该目标 | **0.50 s** | **0.48 s**（持平） |
| 改 `domain_tokenizer_base.hpp` → 全量增量 | **18.0 s / 75 个 CXX 步**（含 BMI 重建 10.6 s + 7 个无 PCH 消费方） | **12.7 s / 83 步**（全部 ccache 命中） |

**结论（如实）**：粒度模块在本仓基线下**不是提速路径**，与全量 import 的结论同源——
ccache 命中已把 include 重编压到 ~0.1s/目标，模块能消除的"改公共头 → 重 parse 全库"
在本仓不存在；反而 BMI 重建（10.6s/tokenizer）成为新的单点热点。隔离收益真实存在
（非模块域零重编、消费方摆脱 PCH 依赖、改 tokenizer 不重编 engine 消费方），但
绝对值被基线追平。落地价值 = **标准符合性**（库提供 C++20 模块消费形态）+
**精确依赖**（`import nn.<域>` 语义化声明依赖）+ **PCH 独立性**；提速不在其中。
若将来出现无 ccache、或多 TU 频繁改公共头的场景，模块路径的隔离收益才会显性化。

### 6.4 复现

```bash
# 模块 ON（默认）：9 个叶子消费者走 import
cmake -B build/mod -G Ninja -DNN_ENABLE_GPU=OFF -DNN_ENABLE_TESTS=ON
cmake --build build/mod --target tokenizer_test nnvocab_test dataset_test dataset_integration_test
# 模块 OFF（回落 include）
cmake -B build/mod-off -G Ninja -DNN_ENABLE_GPU=OFF -DNN_ENABLE_TESTS=ON -DNN_ENABLE_MODULES=OFF
```
