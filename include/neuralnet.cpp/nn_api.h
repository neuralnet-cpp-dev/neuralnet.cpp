// ═══════════════════════════════════════════════════════════════════════════
//  nn_api.h — libnn.so 的稳定 C ABI（Phase 3 link 阶段；docs/internal/34）
//
//  libnn.so 编"具体引擎 + 原语 + 生成器"；本头是消费方的唯一跨边界接口。
//  边界协议（docs/internal/34 §2）：**不透明句柄 + 序列化载荷** ——
//    · 句柄：nnn_engine / nnn_tensor（opaque，谁创建谁释放；tensor 由 engine 领养）
//    · 载荷：ExprSpec 序列化字节（kernel_cache.hpp 的 encode_expr_spec 编码）、
//      裸 float* + dims（Matrix 数据）
//    · STL/模板类型（std::string/vector/expected/dsl 表达式）**永不跨界**
//  C++ 模板面（nn::dsl / Layer 组合）留 header-only 在消费方实例化（nn.hpp）。
//
//  本文件是**最小闭环（Phase 3-A）**：engine / tensor / matmul / eval_expr。
//  纯 C 兼容（无 C++ 依赖），可被 C/C++/ctypes 消费。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nnn_engine_t* nnn_engine;
typedef struct nnn_tensor_t* nnn_tensor;

// 状态码（跨边界错误：STL 的 std::expected 不跨界）
typedef enum nnn_status
{
    NNN_OK = 0,     // 成功
    NNN_ERR = 1,    // 一般错误（具体信息经 nnn_last_error 查询）
    NNN_NO_GPU = 2, // GPU 引擎创建失败（无可用 Vulkan 设备）—— 非致命，可回退 CPU
} nnn_status;

// 精度（对应 nn::Precision 枚举值）
typedef enum nnn_prec
{
    NNN_PREC_F16 = 0,
    NNN_PREC_F32 = 1,
} nnn_prec;

// 最近一次错误的只读文本（libnn 内部缓冲；跨多次调用有效到下次出错；线程本地）。
// 返回 NULL 表示无错误。
const char* nnn_last_error(void);

// ── 引擎生命周期 ─────────────────────────────────────────────────────
// CPU 引擎：确定性、无外部依赖。
nnn_engine nnn_engine_create_cpu(void);
// GPU 引擎：device = 设备名子串（空 = 自动选择，同 NN_VULKAN_DEVICE）；无可用设备
// 返回 NULL（NNN_NO_GPU 已记入 last_error）。消费方可回退 CPU。
nnn_engine nnn_engine_create_gpu(const char* device);
// 释放引擎（连带其领养的全部 tensor 句柄）。
void nnn_engine_destroy(nnn_engine e);

// ── 张量 ────────────────────────────────────────────────────────────
// 创建 (rows, cols) 全零张量（精度 nnn_prec）。
nnn_tensor nnn_tensor_create(nnn_engine e, size_t rows, size_t cols, int prec);
// 从 row-major f32 数据上传建张量（from_matrix）。
nnn_tensor nnn_tensor_from_matrix(nnn_engine e, const float* data, size_t rows, size_t cols,
                                  int prec);
size_t nnn_tensor_rows(nnn_engine e, nnn_tensor t);
size_t nnn_tensor_cols(nnn_engine e, nnn_tensor t);
int nnn_tensor_precision(nnn_engine e, nnn_tensor t);
// 下载到 row-major f32 缓冲（to_matrix）。out 容量须 ≥ rows*cols；返回实际元素数
// 或负值（NNN_ERR）。失败时 out 内容未定义。
nnn_status nnn_tensor_to_matrix(nnn_engine e, nnn_tensor t, float* out, size_t cap,
                                size_t* out_n);
void nnn_tensor_destroy(nnn_engine e, nnn_tensor t);

// ── 原语 ────────────────────────────────────────────────────────────
// matmul：out = A×B（行主序按引擎布局；transA/transB = 是否转置 A/B）。
// out 为新建张量（调用方 nnn_tensor_destroy 释放）。
nnn_status nnn_matmul(nnn_engine e, nnn_tensor A, nnn_tensor B, int transA, int transB,
                      nnn_tensor* out);

// ── 表达式求值（JIT/AOT 融合；docs/internal/33）────────────────────
// spec = encode_expr_spec 编码的 ExprSpec 字节；inputs 为输入张量句柄数组；
// rows/cols = 输出网格形状；prec_sig = 精度签名（0=V0 全 f32）。out 为新建张量。
nnn_status nnn_eval_expr(nnn_engine e, const uint8_t* spec, size_t len,
                         const nnn_tensor* inputs, size_t nin, uint32_t prec_sig, size_t rows,
                         size_t cols, nnn_tensor* out);

#ifdef __cplusplus
}
#endif
