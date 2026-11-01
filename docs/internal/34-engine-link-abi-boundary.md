# ComputeEngine link 阶段：C ABI 边界清单（Phase 3 第一步）

> **读者**：项目开发者（改这个库的人）。本分册定位见 [`README.md`](README.md)；历史归档在 [`../history/history.md`](../history/history.md)。
> **一句话结论**：Phase 3 把 header-only 引擎翻成可 install 的 `libnn.so`（link 阶段，兼解 LGPL 失效与编译慢），成败取决于**先定死 C ABI 边界**——`libnn.so` 只编"具体、稳定"的引擎实现 + 手写原语 + 生成器；模板重（DSL/Layer/Loss/Optimizer 组合）留 header-only 在消费方实例化。边界协议 = **不透明句柄 + 序列化载荷**（STL/模板类型永不跨 `.so`）。
> **状态**：现行（Phase 3-A/B/C 已实施 + 850M 验证：`libnn.so` 门面、`nn_api.h` C ABI、install 布局、LGPL 可替换）。**引擎 .so 化（双引擎 .so + header-only 抽象层转发）为下一步，边界清单见 §10（用户裁定，2026-11-01）**。

## 0. 动机与前置事实

- **LGPL 失效**：header-only 把库代码内联进消费方二进制，没有可替换的 `.so` → LGPL 形同 GPL。解药 = link 阶段（预编译 `.so` + C ABI），让消费方能 link/替换 `libnn.so`（`docs/internal/33` §0.3）。
- **编译慢根因**是 per-TU codegen（PCH 后 `-ftime-trace`：后端 29.1s vs 前端 4.0s），不是头解析。真正提速靠"模板进 header 的只编一次、具体实现编进 `.so`"。
- 前置已交付：生成器 `libnn-kernelgen.so`（C ABI `nnk_generate_spirv`，`kernelgen_api.h`）+ `.nnk` JIT 缓存 + 引擎 JIT miss 路径（`docs/internal/33` Phase 2 ✅）。

## 1. 边界总原则

> **模板/具体分界 = 拆分的唯一成败判据。** 一句话规则：

- **进 `.so`（编译单元，具体类）**：`ComputeEngine` 抽象接口的**纯虚 `*_impl` 族** + 两个具体引擎（`CpuEngine` / `GpuEngine` + `GpuBackend` + Vulkan 后端）+ 手写原语 shader 的 SPIR-V 装载 + `Layer`/`Loss`/`Optimizer`/`Model` 的**非模板虚方法实现** + 生成器 glue。
- **留 header（模板实例化点）**：`nn::dsl` 表达式模板、`Layer` 组合模板（`model.add<T>(...)` 工厂）、`Matrix`/`Tensor` 的访问、`read<T>`/`write<T>` 模板 helper、自定义层/损失/优化器（消费方自己的模板）。

**为什么模板必须留 header**：`Layer::forward/backward` 是虚方法，但 `make_linear_layer`、`model.add<T>`、`nn::dsl::compute` 是模板——消费方实例化自己的层类型与表达式，`.so` 无法预实例化未知模板。边界只能暴露**稳定非模板入口**，模板在消费方就地展开后**调用 `.so` 的稳定原语**。

## 2. 边界协议：不透明句柄 + 序列化载荷

STL（`std::string`/`vector`/`expected`）与模板类型**永不跨 `.so` 边界**（ABI 脆弱：vtable 漂移、`-fno-exceptions` 下 `std::expected` 布局耦合、分配器跨库）。协议：

```
句柄：   void* (nnn_engine_t / nnn_tensor_t / nnn_matrix_t / nnn_model_t)   —— 唯一跨边界身份
载荷：   序列化字节（ExprSpec 已有一份 kernel_cache.hpp 的 encode/decode；
         op 描述符、Matrix 数据、Spec 走各自的 length-prefixed 编码）
错误：   边界函数返回 nnk_status 式枚举 + 可选 last-error 字符串（跨库不安全 std::string）
所有权：谁创建谁释放；tensor/matrix 句柄由 engine 句柄领养，engine 销毁连带回收
```

与 `libnn-kernelgen.so` 的 `nnk_generate_spirv(spec_bytes, prec_sig, out_meta, out_spirv, out_len)` 同一套"字节进/字节出"风格（`docs/internal/33` §3.2 先例）。

## 3. `libnn.so` 边界入口清单（`nn_api.h`，`extern "C"`）

> 这是**候选清单**，动代码前逐项裁决"编不编得进 `.so` / 有没有模板逃逸"。分组：

### 3.1 引擎生命周期
| 入口 | 说明 | 模板? |
|------|------|-------|
| `nnn_engine_create_cpu()` | 返回 CPU 引擎句柄 | 否 |
| `nnn_engine_create_gpu(const char* device)` | 选设备创建 GPU 引擎（`NN_VULKAN_DEVICE` 语义） | 否 |
| `nnn_engine_destroy(nnn_engine_t)` | 释放 | 否 |
| `nnn_engine_describe(nnn_engine_t, char* buf, size_t n)` | `EngineInfo` 序列化文本 | 否 |

### 3.2 张量/矩阵
| 入口 | 说明 | 模板? |
|------|------|-------|
| `nnn_tensor_create(eng, rows, cols, prec)` / `nnn_tensor_destroy` | 声明式创建 | 否 |
| `nnn_tensor_from_matrix(eng, const float* data, rows, cols, prec)` | 上传（`from_matrix`） | 否 |
| `nnn_tensor_to_matrix(eng, tensor, float* out, size_t cap)` | 下载（`to_matrix`） | 否 |
| `nnn_tensor_rows/cols/precision(eng, tensor)` | 形状查询 | 否 |

> `Matrix` 本体（`algebra_matrix.hpp`）是具体类但含内部 `std::vector`/span——**不直接跨界**，用裸 `float* + rows/cols` 或序列化字节。`read<T>`/`write<T>` 模板（`compute_engine.hpp:516/609`）是 CPU 直写便捷层，留 header（内部转裸指针调 `.so` 的 `nnn_*_impl`）。

### 3.3 原语（`*_impl` 族的稳定非模板形参面）
| 入口 | 说明 | 模板? |
|------|------|-------|
| `nnn_matmul(eng, A, B, transA, transB, batch, out)` | `matmul_impl` | 否 |
| `nnn_add_inplace / nnn_scale_inplace / nnn_zero / nnn_clone` | 数据操作 | 否 |
| `nnn_gather_rows / nnn_scatter_add_rows / nnn_rearrange_3d / nnn_transpose` | 数据搬移 | 否 |
| `nnn_row_reduce_sum / nnn_col_reduce_max / nnn_grouped_reduce` | 归约（`reduce_op` 枚举） | 否 |
| `nnn_scan_prefix_outer / nnn_scan_suffix_outer` | RLA/RAPT 扫描 | 否 |
| `nnn_eval_expr(eng, spec_bytes, len, const nnn_tensor_t* in, size_t nin, prec_sig, out)` | **表达式求值**（复用 `encode_expr_spec`）| 否 |
| `nnn_batch_begin / nnn_batch_end / nnn_batch_flush` | 录制窗口 | 否 |
| `nnn_offload_create / nnn_offload_save / nnn_offload_restore` | activation offload | 否 |

### 3.4 模型 forward（可选，Phase 3 后期）
`nnn_model_create_from_spec(spec_bytes)` / `nnn_model_forward(model, tensor_in, tensor_out)` / `nnn_model_destroy`。**风险最大**：`Model` 持 `std::vector<unique_ptr<Layer>>`，Layer 树是模板实例化点——跨界只能句柄化整个 Model，层内模板不得暴露。**Phase 3 先不做 model ABI**，把 scope 压到引擎+原语（§6 分期）。

### 3.5 生成器 + JIT（复用 kernelgen）
`nnn_set_kernelgen_path(const char*)` / 引擎内部把 `.nnk` miss 接到 `nnk_generate_spirv`（Phase 2 已做，边界不新增）。

## 4. 编译单元划分（`libnn.so` 内 vs header）

| 组件 | 现状 | Phase 3 归属 |
|------|------|--------------|
| `ComputeEngine` 接口（纯虚 `*_impl` 族） | header | `.so`（具体引擎实现） |
| `CpuEngine` | header（`compute_cpu_engine.hpp`） | `.so` 编译单元 |
| `GpuEngine` + `GpuBackend` + Vulkan 后端 4 片段 | header | `.so` 编译单元 |
| 手写原语 shader SPIR-V getter（`NN_SPV_GETTER_LIST`） | header 嵌入 | `.so` |
| `Matrix`/`Tensor`/`Scalar` 类型定义 | header | **header**（消费方实例化/布局，`.so` 只经句柄访问） |
| `nn::dsl` 表达式模板 + `expr_spec`/`expr_opt` | header | **header**（模板实例化点） |
| `Layer`/`Loss`/`Optimizer`/`Model` 虚方法实现 | header | `.so`（非模板虚方法）+ header（模板工厂） |
| 生成器 `libnn-kernelgen.so` | `.so` | `.so`（已定） |

> **关键结论**：`.so` 是"具体引擎原语执行体"；header 是"模板组合面"。消费方 `#include <neuralnet.cpp/nn.hpp>` 实例化 DSL/Layer，调用的原语落到 `.so` 的 `nnn_*`。

## 5. install 布局（D8，`GNUInstallDirs`）

```
<prefix>/bin/   cli（mnist/text/tokenizer 各入口）
<prefix>/lib/   libnn.so + libnn-kernelgen.so
<prefix>/include/neuralnet.cpp/*.hpp   + nn_api.h
```
无散落 `.o/.obj`。CMake 角色从"生成 build + 目标散落"演进为"生成 build + 组合成干净安装产物"（`docs/internal/33` §4.3）。

## 6. 分期（把 ABI 风险最大的 model 后移）

| 阶段 | 内容 |
|------|------|
| **A** | `.so` 只编 CPU/GPU 引擎 + 原语 + 张量句柄（§3.1–3.3、§4 引擎行）——**第一步动代码从这开始**，先出 `nnn_engine/tensor/matmul/eval_expr` 最小闭环 ✅（2026-11-01） |
| **B** | 原语面补全（归约/扫描/数据操作进 `nn_api.h`）+ cli 改链 `.so`：**归约（row/col reduce）+ 数据操作（transpose/clone/add/scale/zero）+ 批处理已入 `nn_api.h` 并 850M 验证（2026-11-01，`nn_consumer_test` 全 PASS）**；install 布局含 cli 到 `bin/`（D8）。cli 改链 `.so` 的正解 = **引擎 .so 化（双引擎 .so + header-only 抽象层转发），见 §10** |
| **C** | LGPL 可用性验证（外部消费方 link 可替换 `libnn.so`）✅（2026-11-01，`bench/lgpl_verify.sh`）：① 符号级——`libnn.so` 导出 `nnn_*` C ABI，`nn_consumer_test` 二进制 **0 个引擎实现符号**（引擎全在 `.so`，consumer 只 include C ABI + IR 头）；② 运行时替换——编一个 matmul 输出 ×2 的 patched `libnn.so`，用 `LD_LIBRARY_PATH` 指向它跑**同一** consumer 二进制（不重编）→ matmul 数值 FAIL（行为随 `.so` 变）、eval_expr 不受影响（补丁精准）、恢复原 `.so` 全 PASS。LGPL"可替换库二进制"承诺兑现 |
| **D**（后期，风险高） | model ABI（§3.4）——待 A–C 稳定后单独立项 |

## 7. LGPL 可用性验证口径

- 消费方 `g++ app.cpp -L<prefix>/lib -lnn -o app`（不重编引擎源码），链 `libnn.so`。
- **替换验证**：改 `libnn.so`（如换一个原语实现）后**不重编消费方**，重链/换 `.so` → 消费方行为随新 `.so` 变化 = "可替换库二进制"成立 → LGPL 承诺兑现。
- ctest 全绿（原 header-only 路径不变，双轨：header-only 仍可用，`.so` 是增强）。

> **已落地（2026-11-01，`bench/lgpl_verify.sh`，850M 通过）**：符号级（libnn.so 导出 `nnn_*`、consumer 无引擎实现符号）+ 运行时替换（patched `.so` 使同一 consumer 二进制 matmul 数值变化，恢复原 `.so` 复原）。

## 8. 风险与取舍

| 风险 | 影响 | 对策 |
|------|------|------|
| 模板 vs ABI（最大难点） | DSL/Layer 模板无法藏进 ABI | 边界只暴露稳定原语；模板留 header 就地展开后调 `.so`（§2/§4） |
| STL 跨界 | vtable/分配器/`expected` 布局耦合 | 句柄 + 字节载荷，STL 不跨界（§2） |
| `-fno-exceptions` 下错误 | `Result<T>`= `std::expected` 跨界不稳 | 边界用 `nnk_status` 枚举 + last-error 缓冲 |
| 张量生命周期跨界 | 句柄悬垂/泄漏 | 谁创建谁释放；engine 领养 tensor |
| 双轨维护 | header-only + `.so` 两套要一致 | `.so` 复用同一批 header 实现，无第二份算法；测试双轨跑 |
| 编译慢治标 | `.so` 只消引擎 TU 的重复编译 | 模板（Layer/DSL）仍是消费方大头；README 明示边界 |

## 9. 关联文档

- 前置：`docs/internal/33-kernel-cache-and-engine-decouple.md`（§0.3 LGPL / §4 生成器 `.so` / §10.2 Phase 3 计划 / §10.4 模板 vs ABI）
- 引擎接口：`compute_engine.hpp`（`*_impl` 族）/ `compute_cpu_engine.hpp` / `compute_gpu_engine.hpp` / `backend/compute_vk_backend*.inc`
- 序列化先例：`kernel_cache.hpp` 的 `encode/decode_expr_spec`、`kernelgen_api.h` 的 `nnk_generate_spirv`
- 文件格式：`docs/development/10-kvrecord-format-spec.md`
- 历史归档：`docs/history/history.md`

## 10. 引擎 .so 化：双引擎 .so + header-only 抽象层转发（用户裁定，2026-11-01）

> **目标架构**（§4"B 余"的正解）：把引擎实现真正从 header 抽到**两个独立 `.so`**，库内提供 **C++ 转发层**，让 cli/model/layer 用纯 C++ 接口、**不自己 dlopen** `.so`。本节约"开工前的边界清单 + 已裁定决策"，动代码前先据此裁决。

```
cli / model / layer                    纯 C++，include 抽象层头，用 ComputeEngine 接口
        │
抽象层（header-only C++ 转发）          dlopen 引擎 .so + 句柄/序列化，把 .so 转回 C++ 方法
        ├──────────────┐
libnn-cpu.so         libnn-gpu.so
CpuEngine + CPU 原语   GpuEngine + Vulkan backend + shader + kernelgen
        └── 更底层（backend/原语 shader/生成器）全编进对应 .so ──┘
```

### 10.1 已裁定决策（用户确认）

| # | 决策 | 取值 | 理由 |
|---|------|------|------|
| E1 | 引擎划分 | **两个引擎两个 `.so`**：`libnn-cpu.so`（CpuEngine + CPU 原语）、`libnn-gpu.so`（GpuEngine + Vulkan backend + 原语 shader + kernelgen 生成器） | 解耦：CPU 消费方不拉 Vulkan；各 .so 独立可替换 |
| E2 | 抽象层形态 | **header-only C++ 转发**：`CpuEngine::create()` 等返回转发对象，dlopen 对应 `.so`，方法体转发到 .so C ABI | cli 只 include、不链 .so、不自己 dlopen（用户明确） |
| E3 | Tensor 跨界 | **句柄 + 序列化 C ABI**：抽象层维护 C++ Tensor ↔ .so 句柄映射，方法转成 C ABI（字节/句柄） | 符合 §2"STL/模板不跨界"纪律 + Python 友好（用户明确） |

### 10.2 分步计划（顺序）

1. **抽 `libnn-cpu.so`**：把 `CpuEngine` 实现（`compute_cpu_engine.hpp` 的 `*_impl` + 原语体）编译成独立 `.so`，暴露 CPU 引擎 C ABI（create/tensor/matmul/reduce/eval_expr，复用 §3 的 `nnn_*` 形态）。
2. **抽 `libnn-gpu.so`**：`GpuEngine` + `GpuBackend` + 4 片段 + 原语 shader + kernelgen 生成器编进独立 `.so`，暴露 GPU 引擎 C ABI。
3. **header-only 抽象层**：新增转发类（`CpuEngine`/`GpuEngine` 保留类名，方法体改为 dlopen + 调 .so C ABI + 转回 C++）；Tensor 句柄映射（值语义/拷贝/生命周期）。
4. **cli/model/layer 改接抽象层头**（不链 .so、不 dlopen）。
5. **验证**：cli 训练/推理在 CPU/GPU 全跑通；`nm` 消费方无引擎实现符号；`bench/lgpl_verify.sh` 对两个引擎 .so 分别替换验证；ctest 双轨全绿。

### 10.3 关键风险

- **Tensor 句柄映射**（E3 最大难点）：`Tensor` 是值类型（含 `shared_ptr<GpuBuffer>`/`Matrix`），消费方到处传值/拷贝。抽象层须维护"本地 impl ↔ .so 句柄"双向映射并处理拷贝/析构/生命周期，不得有悬垂或泄漏。
- **性能**：句柄 + 序列化每方法往返有开销；热路径（matmul/训练循环）需要评估是否命中 `.so` 的批量/录制窗口，避免"每个原语一次 IPC 往返"打崩吞吐。
- **双 .so 划分**：kernelgen（JIT 生成器）与 GPU 引擎同 .so 后，`nnk_generate_spirv` 的 dlopen/符号归属要理顺（是引擎内部依赖还是对外入口）。
- **双轨维护**：header-only（现状）与抽象层转发并存期间，两套引擎实现要一致——本方案**最终以 .so 为权威**，header-only 面收敛到抽象层转发（不再各自实例化引擎体）。
