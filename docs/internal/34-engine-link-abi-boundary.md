# ComputeEngine link 阶段：C ABI 边界清单（Phase 3 第一步）

> **读者**：项目开发者（改这个库的人）。本分册定位见 [`README.md`](README.md)；历史归档在 [`../history/history.md`](../history/history.md)。
> **一句话结论**：Phase 3 把 header-only 引擎翻成可 install 的 `libnn.so`（link 阶段，兼解 LGPL 失效与编译慢），成败取决于**先定死 C ABI 边界**——`libnn.so` 只编"具体、稳定"的引擎实现 + 手写原语 + 生成器；模板重（DSL/Layer/Loss/Optimizer 组合）留 header-only 在消费方实例化。边界协议 = **不透明句柄 + 序列化载荷**（STL/模板类型永不跨 `.so`）。
> **状态**：TODO（Phase 3 第一步 = 本边界清单已定稿；`libnn.so` 拆分尚未动代码，动代码前先据此裁决）。

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
| **A** | `.so` 只编 CPU/GPU 引擎 + 原语 + 张量句柄（§3.1–3.3、§4 引擎行）——**第一步动代码从这开始**，先出 `nnn_engine/tensor/matmul/eval_expr` 最小闭环 |
| **B** | cli 改链 `libnn.so`（不再每 TU 全编引擎），install 布局落地 |
| **C** | LGPL 可用性验证（外部消费方 link 可替换 `libnn.so`） |
| **D**（后期，风险高） | model ABI（§3.4）——待 A–C 稳定后单独立项 |

## 7. LGPL 可用性验证口径

- 消费方 `g++ app.cpp -L<prefix>/lib -lnn -o app`（不重编引擎源码），链 `libnn.so`。
- **替换验证**：改 `libnn.so`（如换一个原语实现）后**不重编消费方**，重链/换 `.so` → 消费方行为随新 `.so` 变化 = "可替换库二进制"成立 → LGPL 承诺兑现。
- ctest 全绿（原 header-only 路径不变，双轨：header-only 仍可用，`.so` 是增强）。

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
