// ── nn.kvrecord — C++ 模块：KVRecord v1/v2 容器 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/model_keyvalue_record.hpp>
export module nn.kvrecord;
export import nn.core;
export namespace nn {
    using nn::KeyValueRecord;
    using nn::KvReadAt;
    using nn::kvrec_memory_source;
    namespace detail {
        using nn::detail::append_u32;
        using nn::detail::append_u64;
    }
}
