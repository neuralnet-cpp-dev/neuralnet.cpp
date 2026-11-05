// ── nn.core — C++ 模块：核心域（错误/配置/精度/哈希/文件/线程池/并行）─────
// 「薄接口」模块（research/modules-import-nn 实证）：STL 与库头文件**声明**
// 留在全局模块片段（实体挂全局模块，与 `#include` 消费方同一实体，可混合
// 链接），本模块只做一件事：把 `nn::` 命名空间的公开符号 re-export。
// 约束：宏（NN_TRY / NN_UNWRAP / NN_ASSERT_* 等）**不可导出**——消费者按需
// `#include <nn_error_macros.hpp>` 宏岛头；STL 留在全局片段，消费者自带。
// 导出面 = GMF 文本包含的全部 nn 公开实体（命名空间级 re-export，D7）。
module;
#include <neuralnet.cpp/core_assert.hpp>
#include <neuralnet.cpp/core_config.hpp>
#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/core_observer_ptr.hpp>
#include <neuralnet.cpp/core_file.hpp>
#include <neuralnet.cpp/core_parallel.hpp>
#include <neuralnet.cpp/core_sha256.hpp>
#include <neuralnet.cpp/core_threadpool.hpp>
#include <neuralnet.cpp/precision.hpp>
export module nn.core;

export using namespace nn;
