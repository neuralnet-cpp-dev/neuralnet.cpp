// ── nn.core — C++ 模块：核心域（错误/配置/精度/哈希/文件/线程池/并行）─────
// 「薄接口」模块（research/modules-import-nn 实证）：STL 与库头文件**声明**
// 留在全局模块片段（实体挂全局模块，与 `#include` 消费方同一实体，可混合
// 链接），本模块只做一件事：把 `nn::` 命名空间的公开符号 re-export。
// 约束：宏（NN_TRY / NN_UNWRAP / NN_ASSERT_* / NN_VECTORIZE_PRAGMA 等）
// **不可导出**——消费者按需 `#include <neuralnet.cpp/core_errors.hpp>` 等
// 宏头（双轨）；STL 留在全局片段，消费者自带用到的 STL 头。
// 导出清单 = 下列头文件在 `namespace nn` 内的公开 API（detail 一律不导出）。
module;
#include <neuralnet.cpp/core_assert.hpp>
#include <neuralnet.cpp/core_config.hpp>
#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/core_file.hpp>
#include <neuralnet.cpp/core_parallel.hpp>
#include <neuralnet.cpp/core_sha256.hpp>
#include <neuralnet.cpp/core_threadpool.hpp>
#include <neuralnet.cpp/precision.hpp>
export module nn.core;

export namespace nn {
    // core_errors.hpp
    using nn::Error;
    using nn::Result;
    using nn::parse_number;
    // core_config.hpp
    using nn::Scalar;
    using nn::BLOCK_SIZE;
    using nn::EPSILON;
    using nn::PARALLEL_THRESHOLD;
    // precision.hpp
    using nn::Precision;
    using nn::precision_rank;
    using nn::precision_name;
    using nn::max_precision;
    using nn::precision_tag;
    using nn::precision_from_tag;
    using nn::f16;
    using nn::f32;
    using nn::Elem;
    using nn::elem;
    using nn::acc;
    using nn::PrecisionProfile;
    using nn::profile_f32;
    using nn::profile_master_weights;
    using nn::profile_f16;
    using nn::profile_t1_58;
    using nn::profile_all_f16;
    using nn::is_profile_f32;
    // core_sha256.hpp（Sha256State / sha256_* 内部件在 detail，不导出）
    using nn::sha256_hex;
    // core_file.hpp
    using nn::load_text_file;
    using nn::sha256_file_prefix;
    using nn::sha256_file_hex;
    using nn::write_pod;
    using nn::read_pod;
    using nn::write_pod_span;
    using nn::read_pod_span;
    // core_threadpool.hpp
    using nn::ThreadPool;
    using nn::global_thread_pool;
    // core_parallel.hpp
    using nn::for_each;
    using nn::transform;
    using nn::transform_reduce;
    using nn::parallel_for_blocks;
    using nn::parallel_for_samples;
}
