// ── nn.kvrecord — C++ 模块：KVRecord v1/v2 容器 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）。
// 实体在 kvrec submodule（namespace kvrec）——经 export using 使 import 方
// 直接可见 kvrec::*（与 nn.core 对 exstl 同一模式）；**不在 nn 内做别名**。
// 宏不可导出（消费方 include <exstl/error_macros.hpp>）。
module;
#include <kvrec/kvrecord.hpp>
export module nn.kvrecord;
export import nn.core;

export using kvrec::KeyValueRecord;
export using kvrec::KvReadAt;
export using kvrec::kvrec_memory_source;
export namespace kvrec::detail
{
    using kvrec::detail::append_u32;
    using kvrec::detail::append_u64;
}
