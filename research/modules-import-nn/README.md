# modules-import-nn — `import nn` vs `#include <nn.hpp>` 编译速度

> **读者**：项目开发者——评估"用 C++20 模块重写库加载路径"能否提升编译速度、并符合高版本标准。
> **结论（分两层）**：① **全量 `import nn` 否决**——热路径(0.19s)比现行 ccache 命中(0.01s)慢约 19×，
> 86 MB BMI + 宏不可导出 + STL 无法干净进模块三重硬约束，实测不采纳。② **粒度导入是方向**——
> 改库加载路径的动机不止提速，还有**标准符合性**(库应是 C++20/26 模块形态而非纯 `#include`)；
> 真正的提速来自**按功能域拆窄模块**(`nn.tokenizer` 闭包仅 12 头 vs 全库 200+ 头)、消费者只
> `import` 所需域，而非全量塞进一个巨型 BMI。粒度拆分**本 spike 未测**，留作后续。
> **状态**：现行（2026-11-01 实测，clang 24 / Linux / -O3 / 纯 CPU）。全量 import 不进主线；粒度导入待后续 spike。

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

**本 spike 未测粒度拆分**(只测了全量 import vs include)。粒度模块的 BMI 体量/建造/隔离收益
留作**后续独立 spike**——落地前需按功能域划模块边界、处理跨域符号 re-export、并实测
「粒度 import vs 全量 include+ccache」在真实多 TU 消费下的重编风暴差异。

**当前主线不变**:`#include <nn.hpp>` + 共享 PCH(`nn_pch`)+ ccache。
粒度模块若立项,它是**增量接入**(下游逐步改用 `import nn.<domain>`),不是一次性替换。
