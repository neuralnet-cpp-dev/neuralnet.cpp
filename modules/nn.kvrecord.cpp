// ── nn.kvrecord — C++ 模块：KVRecord（v1/v2 键值记录）─────────────────────
// 薄接口模块（见 nn.core.cpp 头注释）：依赖域经 `export import` re-export，
// 消费方只 `import nn.kvrecord;` 即可连带拿到 nn.core 的公开符号。
module;
#include <neuralnet.cpp/model_keyvalue_record.hpp>
export module nn.kvrecord;
export import nn.core;

export namespace nn {
    using nn::KeyValueRecord;
    using nn::kvrec_memory_source;
}
