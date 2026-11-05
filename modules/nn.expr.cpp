// ── nn.expr — C++ 模块：表达式 DSL / IR / 张量视图 ──────────
// 薄接口模块：GMF 文本包含域头（实体挂全局模块，与 #include 消费方可混链）；
// 导出 = `export namespace nn { … }` 逐符号（D7 命名空间级导出已证伪）。
// ⚠ 嵌套名（nn::cli::X / nn::dsl::X / nn::detail::X）必须**镜像原命名空间**
// 导出——`using nn::cli::X;` 写在 nn 块里会被 clang 扁平引入为 nn::X，原限定
// 路径不可见（实测）。宏不可导出（消费方 include <nn_error_macros.hpp>）。
module;
#include <neuralnet.cpp/expr_dsl.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/expr_opt.hpp>
#include <neuralnet.cpp/expr_graph.hpp>
#include <neuralnet.cpp/expr_fold.hpp>
#include <neuralnet.cpp/expr_registry.hpp>
#include <neuralnet.cpp/compute_tensor.hpp>
#include <neuralnet.cpp/tensor_schema.hpp>
#include <neuralnet.cpp/named_tensor_set.hpp>
export module nn.expr;
export import nn.core;
export namespace nn {
    using nn::detail::float_to_half_bits;
    using nn::detail::half_bits_to_float;
    using nn::ExprOp;
    using nn::ExprOperandKind;
    using nn::ExprSpec;
    using nn::FoldSpec;
    using nn::MatmulSpec;
    using nn::ExprOp;
    using nn::ExprOperandKind;
    using nn::ExprSpec;
    using nn::FoldSpec;
    using nn::MatmulSpec;
    using nn::ComputeEngine;
    using nn::Device;
    using nn::DomainId;
    using nn::EXPR_FOLD_BLOCK;
    using nn::EXPR_MAX_CONSTS;
    using nn::EXPR_MAX_INPUTS;
    using nn::EXPR_MAX_INSTRS;
    using nn::EXPR_MAX_REGS;
    using nn::EXPR_PREC_DISPATCH_SUFFIX;
    using nn::EXPR_PREC_SIG_DISPATCH;
    using nn::Error;
    using nn::ExprGraph;
    using nn::ExprInstr;
    using nn::ExprOp;
    using nn::ExprOperandKind;
    using nn::ExprSpec;
    using nn::ExprView;
    using nn::ExprViewKind;
    using nn::FieldHandle;
    using nn::FoldSpec;
    using nn::FusedKernel;
    using nn::GradientBundle;
    using nn::MatmulSpec;
    using nn::Matrix;
    using nn::MatrixT;
    using nn::NamedTensor;
    using nn::NamedTensorSet;
    using nn::PARALLEL_THRESHOLD;
    using nn::Precision;
    using nn::PrecisionProfile;
    using nn::Result;
    using nn::Scalar;
    using nn::Tensor;
    using nn::TensorFieldSpec;
    using nn::TensorPlacement;
    using nn::TensorRef;
    using nn::TensorSchema;
    using nn::TensorShapeSpec;
    using nn::ThreadPool;
    using nn::VecAccSpec;
    using nn::canonicalize_expr_spec;
    using nn::check_precision_supported;
    using nn::decode_tensor_name_list;
    using nn::decode_tensor_schema;
    using nn::dst_order_constraints;
    using nn::elem;
    using nn::expr_op_is_reduce;
    using nn::expr_spec_equal;
    using nn::expr_spec_has_matmul;
    using nn::expr_spec_key;
    using nn::expr_spec_reduce_axis;
    using nn::expr_spec_runtime_matmul_trans;
    using nn::f16;
    using nn::for_each;
    using nn::fuse_expr_graph;
    using nn::global_thread_pool;
    using nn::is_profile_f32;
    using nn::kHostDomain;
    using nn::max_precision;
    using nn::parallel_for_blocks;
    using nn::parallel_for_samples;
    using nn::parse_number;
    using nn::precision_bytes;
    using nn::precision_from_tag;
    using nn::precision_has_order;
    using nn::precision_name;
    using nn::precision_rank;
    using nn::precision_tag;
    using nn::profile_all_f16;
    using nn::profile_f16;
    using nn::profile_f32;
    using nn::profile_master_weights;
    using nn::profile_t1_58;
    using nn::tensor_field;
    using nn::transform;
    using nn::transform_reduce;
    using nn::validate_dst_order;
    using nn::validate_expr_spec;
    namespace dsl {
        using nn::dsl::row_mod;
        using nn::dsl::tanh;
        using nn::dsl::log;
        using nn::dsl::sqrt;
        using nn::dsl::abs;
        using nn::dsl::min;
using nn::dsl::operator+;
using nn::dsl::operator-;
using nn::dsl::operator*;
using nn::dsl::operator/;
using nn::dsl::operator>;
using nn::dsl::operator<;
using nn::dsl::operator>=;
using nn::dsl::operator<=;
using nn::dsl::operator==;
using nn::dsl::operator!=;
        using nn::dsl::ConstLeaf;
        using nn::dsl::GroupedReduceMaxRef;
        using nn::dsl::GroupedReduceSumRef;
        using nn::dsl::TensorRef;
        using nn::dsl::col_broadcast;
        using nn::dsl::col_reduce_max;
        using nn::dsl::col_reduce_sum;
        using nn::dsl::compute;
        using nn::dsl::compute_into;
        using nn::dsl::compute_reduce;
        using nn::dsl::exp;
        using nn::dsl::grouped_reduce_max;
        using nn::dsl::leaf;
        using nn::dsl::matmul;
        using nn::dsl::matmul_kv;
        using nn::dsl::max;
        using nn::dsl::row_broadcast;
        using nn::dsl::row_gather;
        using nn::dsl::row_reduce_max;
        using nn::dsl::row_reduce_sum;
        using nn::dsl::rparam;
        using nn::dsl::select;
        using nn::dsl::to_expr_spec;
    }
    namespace expr {
        using nn::expr::cst;
        using nn::expr::input;
        using nn::expr::linear;
        using nn::expr::make_fold_logsumexp;
        using nn::expr::make_fold_rowmax;
        using nn::expr::make_fold_rowsum;
        using nn::expr::make_fold_softmax_denom;
        using nn::expr::matmul_op;
        using nn::expr::reduce;
        using nn::expr::reg;
        using nn::expr::trans;
    }
    namespace fused {
        using nn::fused::ExprRegistry;
    }
    namespace i8_pack {
        }
    namespace detail {
        using nn::detail::schema_put_string;
        using nn::detail::schema_put_u32;
    }
    namespace dsl {
        using nn::dsl::compute;
        using nn::dsl::eval_cpu;
        using nn::dsl::grouped_reduce_sum;
        using nn::dsl::matmul;
        using nn::dsl::relu;
        using nn::dsl::rotate_half;
        using nn::dsl::row;
        using nn::dsl::row_access;
    }
    namespace expr {
        using nn::expr::rotate_half;
        using nn::expr::row_mod;
    }
}
