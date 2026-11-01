// ═══════════════════════════════════════════════════════════════════════════
//  kernelgen.cpp — libnn-kernelgen.so：运行期 JIT 算子生成器
//
//  引擎在 .nnk 缓存 miss 时调用本 .so 现编 SPIR-V：
//    序列化 ExprSpec → canonicalize → GlslEmitter 生成 GLSL → shaderc(glslang) → SPIR-V。
//  链接 shaderc_shared（内嵌 glslang），版本化（nnk_generator_version = K 的版本组件）。
//  C ABI 见 kernelgen_api.h。
//
//  与构建期 glslc 同源（glslc 就是 shaderc 的命令行前端）→ 运行期 JIT 与
//  构建期 AOT 产生的 SPIR-V 语义一致。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include <shaderc/shaderc.h>

#include <neuralnet.cpp/expr_emitter.hpp>
#include <neuralnet.cpp/expr_glsl_gen.hpp>   // 注册 GlslEmitter（emitter_registry::make("glsl")）
#include <neuralnet.cpp/expr_opt.hpp>        // canonicalize_expr_spec / validate_expr_spec
#include <neuralnet.cpp/kernel_cache.hpp>    // decode_expr_spec / kKernelCacheGeneratorVersion
#include <neuralnet.cpp/kernelgen_api.h>

extern "C"
{

const char* nnk_generator_version(void)
{
    return nn::fused::kKernelCacheGeneratorVersion;
}

void nnk_free_spirv(uint8_t* spirv)
{
    std::free(spirv);
}

nnk_status nnk_generate_spirv(const std::uint8_t* spec_bytes, std::size_t spec_len,
                              std::uint32_t prec_sig, nnk_kernel_meta* out_meta,
                              std::uint8_t** out_spirv, std::size_t* out_len)
{
    *out_spirv = nullptr;
    *out_len = 0;
    if (spec_bytes == nullptr || spec_len == 0) return NNK_BAD_SPEC;

    // 1) 序列化 ExprSpec → canonical 结构
    auto spec_r = nn::fused::decode_expr_spec(spec_bytes, spec_len);
    if (!spec_r) return NNK_BAD_SPEC;
    const nn::ExprSpec canon = nn::canonicalize_expr_spec(*spec_r);

    // 2) 运行期 kernel 元数据（与构建期 FusedShader / 后端 FusedKernel 同源）：
    //    由 canonical spec + 生成形态推导，供引擎懒建 pipeline。单一来源 =
    //    nn::fused_kernel_meta_of（expr_glsl_gen.hpp，引擎侧同一函数）→ 两侧
    //    永不漂移（vec_width 失配 = dispatch 宽度错 = 静默算 1/4）。
    if (out_meta)
    {
        const nn::FusedKernelMeta m = nn::fused_kernel_meta_of(canon);
        out_meta->reduce_axis = m.reduce_axis;
        out_meta->has_matmul = m.has_matmul;
        out_meta->view_param_count = m.view_param_count;
        out_meta->rparam_count = m.rparam_count;
        out_meta->vec_width = m.vec_width;
        out_meta->fold_out = m.fold_out;
        out_meta->fold_per_thread_row = m.fold_per_thread_row;
        out_meta->pc_base_fixed = m.pc_base_fixed;
    }

    // 3) GlslEmitter 生成 GLSL（归约走 generate_reduce，逐元素走 generate）
    auto emitter = nn::emitter_registry::make("glsl");
    if (!emitter) return NNK_COMPILE_FAIL;
    const int raxis = nn::expr_spec_reduce_axis(canon);
    const std::string glsl = (raxis >= 0)
                                 ? emitter->generate_reduce("jit_kernel", canon, prec_sig)
                                 : emitter->generate("jit_kernel", canon, prec_sig, /*alu=*/false);
    if (glsl.empty()) return NNK_COMPILE_FAIL;

    // 4) shaderc 编译 GLSL compute → SPIR-V（默认目标环境 = Vulkan 1.0）
    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    if (compiler == nullptr) return NNK_COMPILE_FAIL;
    shaderc_compile_options_t opts = shaderc_compile_options_initialize();
    shaderc_compilation_result_t res = shaderc_compile_into_spv(
        compiler, glsl.data(), glsl.size(), shaderc_compute_shader, "jit", "main", opts);
    shaderc_compile_options_release(opts);
    shaderc_compiler_release(compiler);
    if (res == nullptr ||
        shaderc_result_get_compilation_status(res) != shaderc_compilation_status_success)
    {
        if (res != nullptr) shaderc_result_release(res);
        return NNK_COMPILE_FAIL;
    }
    const std::size_t len = shaderc_result_get_length(res);
    const char* data = shaderc_result_get_bytes(res);
    std::uint8_t* buf = static_cast<std::uint8_t*>(std::malloc(len));
    if (buf == nullptr)
    {
        shaderc_result_release(res);
        return NNK_COMPILE_FAIL;
    }
    std::memcpy(buf, data, len);
    shaderc_result_release(res);
    *out_spirv = buf;
    *out_len = len;
    return NNK_OK;
}

} // extern "C"
