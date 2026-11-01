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

// 生成器版本号（应为 .nnk 缓存 K 的版本组件；引擎用它校验一致性）。
const char* nnk_generator_version(void);

// 生成 SPIR-V：
//   spec_bytes / spec_len : 序列化 ExprSpec（kernel_cache.hpp 的 encode_expr_spec 载荷）
//   prec_sig              : 所需精度签名（0=V0 全 f32；EXPR_PREC_SIG_DISPATCH=分派；
//                           其余 = 静态签名位图）
//   out_spirv / out_len   : 输出的 SPIR-V 字节（nnk 内部 malloc，调用方必须 free）
// 返回 NNK_OK 时 out_spirv 非空；失败时 out_spirv=nullptr。
nnk_status nnk_generate_spirv(const uint8_t* spec_bytes, size_t spec_len,
                              uint32_t prec_sig, uint8_t** out_spirv, size_t* out_len);

// 释放 nnk_generate_spirv 返回的 SPIR-V 缓冲。
void nnk_free_spirv(uint8_t* spirv);

#ifdef __cplusplus
}
#endif
