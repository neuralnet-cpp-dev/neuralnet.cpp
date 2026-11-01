// ═══════════════════════════════════════════════════════════════════════════
//  kernelgen_api.h — libnn-kernelgen.so 的稳定 C ABI（运行期 JIT 算子生成器）
//
//  生成器 .so 内嵌 GlslEmitter（IR → GLSL）+ shaderc/glslang（GLSL → SPIR-V）。
//  引擎在 .nnk 缓存 miss 时调用本接口现编 SPIR-V 并追加进缓存。
//  设计见 docs/internal/33-kernel-cache-and-engine-decouple.md §3.2/§4。
//
//  纯 C 兼容头（无 C++ 依赖），可被 C/C++/ctypes 消费。
// ═══════════════════════════════════════════════════════════════════════════
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nnk_status
{
    NNK_OK = 0,        // 成功
    NNK_BAD_SPEC = 1,  // 序列化 ExprSpec 损坏/不可解析
    NNK_COMPILE_FAIL = 2,  // GLSL 生成或 SPIR-V 编译失败
} nnk_status;

// 生成结果的运行期 kernel 元数据（与构建期 FusedShader / 后端 FusedKernel
// 同源，见 fused_generate.hpp 的 fused_vec_width_for 与
// compute_vk_backend_destroy.inc 的 pc_base_fixed 推导）。引擎据它建 pipeline。
typedef struct nnk_kernel_meta
{
    int32_t reduce_axis;       // -1=逐元素, 0=行归约, 1=列归约
    uint32_t has_matmul;       // 1=含前置 matmul 段（PC 多 rows + mm_k）
    uint32_t view_param_count; // vp push constant 槽数
    uint32_t rparam_count;     // rp push constant 槽数
    uint32_t vec_width;        // 每线程元素数（1=标量, 4=vec4）
    uint32_t fold_out;         // fold 输出列数（0=非 fold）
    uint32_t fold_per_thread_row; // 1 → v1 标量 fold；0 → v2 双域 fold
    uint32_t pc_base_fixed;    // 固定头 uint 槽数（不含 prec/vp/常量池）
} nnk_kernel_meta;

// 生成器版本号（应为 .nnk 缓存 K 的版本组件；引擎用它校验一致性）。
const char* nnk_generator_version(void);

// 生成 SPIR-V：
//   spec_bytes / spec_len : 序列化 ExprSpec（kernel_cache.hpp 的 encode_expr_spec 载荷）
//   prec_sig              : 所需精度签名（0=V0 全 f32；EXPR_PREC_SIG_DISPATCH=分派；
//                           其余 = 静态签名位图）
//   out_meta              : 输出 kernel 元数据（可空；NNK_OK 时填充）
//   out_spirv / out_len   : 输出的 SPIR-V 字节（nnk 内部 malloc，调用方必须 free）
// 返回 NNK_OK 时 out_spirv 非空；失败时 out_spirv=nullptr。
nnk_status nnk_generate_spirv(const uint8_t* spec_bytes, size_t spec_len,
                              uint32_t prec_sig, nnk_kernel_meta* out_meta,
                              uint8_t** out_spirv, size_t* out_len);

// 释放 nnk_generate_spirv 返回的 SPIR-V 缓冲。
void nnk_free_spirv(uint8_t* spirv);

#ifdef __cplusplus
}
#endif
