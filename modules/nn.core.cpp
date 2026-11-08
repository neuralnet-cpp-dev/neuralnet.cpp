// ── nn.core — C++ 模块：核心域（错误/配置/精度/哈希/文件/线程池/并行） ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/core_assert.hpp>
#include <neuralnet.cpp/core_config.hpp>
#include <neuralnet.cpp/core_env.hpp>
#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/core_observer_ptr.hpp>
#include <neuralnet.cpp/core_file.hpp>
#include <neuralnet.cpp/core_parallel.hpp>
#include <neuralnet.cpp/core_sha256.hpp>
#include <neuralnet.cpp/core_threadpool.hpp>
#include <neuralnet.cpp/precision.hpp>
export module nn.core;
export namespace nn {
    using nn::BLOCK_SIZE;
    using nn::EPSILON;
    using nn::Elem;
    using nn::Error;
    using nn::PARALLEL_THRESHOLD;
    using nn::Precision;
    using nn::PrecisionProfile;
    using nn::Result;
    using nn::Scalar;
    using nn::ThreadPool;
    using nn::acc;
    using nn::elem;
    using nn::f16;
    using nn::f32;
    using nn::for_each;
    using nn::global_thread_pool;
    using nn::is_profile_f32;
    using nn::load_text_file;
    using nn::max_precision;
    using nn::parallel_for_blocks;
    using nn::parallel_for_samples;
    using nn::parse_number;
    using nn::precision_from_tag;
    using nn::precision_name;
    using nn::precision_rank;
    using nn::precision_tag;
    using nn::profile_all_f16;
    using nn::profile_f16;
    using nn::profile_f32;
    using nn::profile_master_weights;
    using nn::profile_t1_58;
    using nn::read_pod;
    using nn::read_pod_span;
    using nn::sha256_file_hex;
    using nn::sha256_file_prefix;
    using nn::sha256_hex;
    using nn::transform;
    using nn::transform_reduce;
    using nn::write_pod;
    using nn::write_pod_span;
    namespace detail {
        using nn::detail::Sha256State;
        using nn::detail::env_flag_raw;
        using nn::detail::env_int_raw;
        using nn::detail::env_string_raw;
        using nn::detail::float_to_half_bits;
        using nn::detail::half_bits_to_float;
        using nn::detail::probe_flag_raw;
        using nn::detail::probe_string_raw;
        using nn::detail::sha256_digest;
        using nn::detail::sha256_finalize;
        using nn::detail::sha256_update;
    }
}
