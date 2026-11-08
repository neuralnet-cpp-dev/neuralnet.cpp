// ── nn.engine — C++ 模块：计算引擎抽象层 + 后端 + 代数/精度原语 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/compute_engine.hpp>
#include <neuralnet.cpp/algebra_matrix.hpp>
#include <neuralnet.cpp/backend/compute_vk_backend.hpp>
#include <neuralnet.cpp/ternary_pack.hpp>
#include <neuralnet.cpp/int8_pack.hpp>
#include <neuralnet.cpp/core_profile.hpp>
#include <neuralnet.cpp/kernel_cache.hpp>
#include <neuralnet.cpp/kernelgen_api.h>
export module nn.engine;
export import nn.core;
export namespace nn {
    namespace trit_pack {
        using nn::trit_pack::code_of;
        using nn::trit_pack::dequant_row;
        using nn::trit_pack::kBase3PerByte;
        using nn::trit_pack::at;
        using nn::trit_pack::at_base3;
        using nn::trit_pack::base3_bytes_per_row;
        using nn::trit_pack::bytes_per_row;
        using nn::trit_pack::latent_row_scale;
        using nn::trit_pack::pack_row;
        using nn::trit_pack::pack_row_base3;
        using nn::trit_pack::unpack_row_base3;
        using nn::trit_pack::value_of;
        using nn::trit_pack::words_per_row;
        using nn::trit_pack::kBits;
        using nn::trit_pack::kPerByte;
        using nn::trit_pack::kPerWord;
        using nn::trit_pack::kMask;
    }
    using nn::create_cpu_engine;
    using nn::ExprSegment;
    using nn::InferenceScope;
    using nn::InitSpec;
    using nn::KeyValueRecord;
    using nn::create_gpu_engine;
    using nn::download_span;
    using nn::download_vector;
    using nn::kInitSeed;
    using nn::kvrec_memory_source;
    using nn::last_error;
    using nn::upload_span;
    namespace fused {
        using nn::fused::KernelCache;
        using nn::fused::decode_expr_spec;
        using nn::fused::encode_expr_spec;
        using nn::fused::encode_kernel_cache;
        using nn::fused::kKernelCacheGeneratorVersion;
        using nn::fused::kernel_cache_key;
        using nn::fused::kernel_cache_path;
        using nn::fused::kernel_cache_path_from_env;
    }
    namespace i8_pack {
        using nn::i8_pack::pack_row;
        using nn::i8_pack::quantize;
        using nn::i8_pack::words_per_row;
    }
    namespace i8_pack {
        using nn::i8_pack::at;
        using nn::i8_pack::pack_row;
        using nn::i8_pack::words_per_row;
    }
    namespace prof {
        using nn::prof::dump;
        using nn::prof::enabled;
    }
    namespace trit_pack {
        using nn::trit_pack::at_base3;
        using nn::trit_pack::base3_bytes_per_row;
        using nn::trit_pack::bytes_per_row;
        using nn::trit_pack::latent_row_scale;
        using nn::trit_pack::pack_row;
        using nn::trit_pack::pack_row_base3;
        using nn::trit_pack::unpack_row_base3;
        using nn::trit_pack::value_of;
        using nn::trit_pack::words_per_row;
    }
}
