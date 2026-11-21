#pragma once

// ── rl_experience.hpp — 可扩展经验记录与 advantage 变换（docs/internal/31 RL-1a）
//
// 形状约定：转移字段使用 `(seq, batch)`，row = 时间、col = 独立轨迹。
// `T = seq * batch` 仍可由调用方按 batch-major 展平；经验 schema 不把数据
// 本体绑定到 TextTrainer、Dataset 或某个具体 RL 算法。
//
// 数值变换只使用 DSL/引擎原语。GAE / return 的逆序递推由宿主调度步序，
// 每步张量计算与递推状态留在设备侧；归一化只在边界回读三个统计标量。
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compute_engine.hpp"
#include "expr_dsl.hpp"
#include "named_tensor_set.hpp"
#include "tensor_schema.hpp"

namespace nn
{

class TensorRecord
{
private:
    NamedTensorSet values_;

    explicit TensorRecord(NamedTensorSet values) : values_(std::move(values)) {}

public:
    TensorRecord() = default;
    TensorRecord(const TensorRecord&) = delete;
    TensorRecord& operator=(const TensorRecord&) = delete;
    TensorRecord(TensorRecord&&) noexcept = default;
    TensorRecord& operator=(TensorRecord&&) noexcept = default;
    ~TensorRecord() = default;

    [[nodiscard]] static exstl::Result<TensorRecord> create(TensorSchema schema,
                                                     std::vector<NamedTensor> values)
    {
        EX_TRY(set, NamedTensorSet::create(std::move(schema), std::move(values)));
        return TensorRecord(std::move(*set));
    }

    [[nodiscard]] const TensorSchema& schema() const noexcept { return values_.schema(); }
    [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
    [[nodiscard]] exstl::Result<FieldHandle> field(std::string_view name) const
    { return values_.field(name); }
    [[nodiscard]] exstl::Result<FieldHandle> optional_field(std::string_view name) const
    { return values_.optional_field(name); }
    [[nodiscard]] exstl::Result<const Tensor*> require(const FieldHandle& handle) const
    { return values_.require(handle); }
    [[nodiscard]] exstl::Result<const Tensor*> require(std::string_view name) const
    { return values_.require(name); }
    [[nodiscard]] const Tensor* find(const FieldHandle& handle) const noexcept
    { return values_.find(handle); }
    [[nodiscard]] const Tensor* find(std::string_view name) const noexcept
    { return values_.find(name); }
};

class ExperienceSchema
{
private:
    TensorSchema schema_;
    std::optional<FieldHandle> action_;
    std::optional<FieldHandle> old_log_prob_;
    std::optional<FieldHandle> advantage_;
    std::optional<FieldHandle> returns_;
    std::optional<FieldHandle> old_value_;
    std::optional<FieldHandle> action_mask_;
    std::optional<FieldHandle> value_mask_;
    std::optional<FieldHandle> reward_;
    std::optional<FieldHandle> done_;
    std::optional<FieldHandle> group_id_;

    static void resolve_(const TensorSchema& schema, std::string_view name,
                         std::optional<FieldHandle>& out)
    {
        auto h = EX_UNWRAP(schema.optional_field(name));
        if (h.valid()) out = h;
    }

    ExperienceSchema(TensorSchema schema, std::vector<std::string>) : schema_(std::move(schema))
    {
        resolve_(schema_, "action", action_);
        resolve_(schema_, "old_log_prob", old_log_prob_);
        resolve_(schema_, "advantage", advantage_);
        resolve_(schema_, "return", returns_);
        resolve_(schema_, "old_value", old_value_);
        resolve_(schema_, "action_mask", action_mask_);
        resolve_(schema_, "value_mask", value_mask_);
        resolve_(schema_, "reward", reward_);
        resolve_(schema_, "done", done_);
        resolve_(schema_, "group_id", group_id_);
    }

public:
    ExperienceSchema() = default;
    ExperienceSchema(const ExperienceSchema&) = default;
    ExperienceSchema& operator=(const ExperienceSchema&) = default;
    ExperienceSchema(ExperienceSchema&&) noexcept = default;
    ExperienceSchema& operator=(ExperienceSchema&&) noexcept = default;
    ~ExperienceSchema() = default;

    // 标准字段先全部登记为可选，再按 required 列表收紧；custom 字段追加在
    // 标准字段之后。required 中的未知名称 -> 明确错误。
    [[nodiscard]] static exstl::Result<ExperienceSchema> create(std::vector<std::string> required = {},
                                                         std::vector<TensorFieldSpec> custom = {})
    {
        std::vector<TensorFieldSpec> fields{
            tensor_field("action", {}, false),     tensor_field("old_log_prob", {}, false),
            tensor_field("advantage", {}, false),  tensor_field("return", {}, false),
            tensor_field("old_value", {}, false),  tensor_field("action_mask", {}, false),
            tensor_field("value_mask", {}, false), tensor_field("reward", {}, false),
            tensor_field("done", {}, false),       tensor_field("group_id", {}, false),
        };
        for (const std::string& name : required)
        {
            bool found = false;
            for (TensorFieldSpec& field : fields)
            {
                if (field.name == name)
                {
                    field.required = true;
                    found = true;
                    break;
                }
            }
            if (!found)
                EX_FAIL("ExperienceSchema::create: required 含未知标准字段 '" + name +
                        "'（自定义字段请在定义中自行设 required）");
        }
        fields.insert(fields.end(), custom.begin(), custom.end());
        EX_TRY(schema, TensorSchema::create("experience", std::move(fields)));
        return ExperienceSchema(std::move(*schema), std::move(required));
    }

    [[nodiscard]] const TensorSchema& schema() const noexcept { return schema_; }
    [[nodiscard]] std::optional<FieldHandle> action() const noexcept { return action_; }
    [[nodiscard]] std::optional<FieldHandle> old_log_prob() const noexcept { return old_log_prob_; }
    [[nodiscard]] std::optional<FieldHandle> advantage() const noexcept { return advantage_; }
    [[nodiscard]] std::optional<FieldHandle> returns() const noexcept { return returns_; }
    [[nodiscard]] std::optional<FieldHandle> old_value() const noexcept { return old_value_; }
    [[nodiscard]] std::optional<FieldHandle> action_mask() const noexcept { return action_mask_; }
    [[nodiscard]] std::optional<FieldHandle> value_mask() const noexcept { return value_mask_; }
    [[nodiscard]] std::optional<FieldHandle> reward() const noexcept { return reward_; }
    [[nodiscard]] std::optional<FieldHandle> done() const noexcept { return done_; }
    [[nodiscard]] std::optional<FieldHandle> group_id() const noexcept { return group_id_; }
};

class PolicyBatchView
{
private:
    const TensorRecord* record_ = nullptr;
    FieldHandle action_{};
    FieldHandle action_mask_{};
    std::optional<FieldHandle> old_log_prob_;
    std::optional<FieldHandle> advantage_;
    std::optional<FieldHandle> returns_;
    std::optional<FieldHandle> old_value_;
    std::optional<FieldHandle> value_mask_;
    std::optional<FieldHandle> group_id_;

    PolicyBatchView(const TensorRecord& record) : record_(&record) {}

public:
    PolicyBatchView() = default;
    PolicyBatchView(const PolicyBatchView&) = default;
    PolicyBatchView& operator=(const PolicyBatchView&) = default;
    PolicyBatchView(PolicyBatchView&&) noexcept = default;
    PolicyBatchView& operator=(PolicyBatchView&&) noexcept = default;
    ~PolicyBatchView() = default;

    [[nodiscard]] static exstl::Result<PolicyBatchView> create(const TensorRecord& record)
    {
        auto action = record.field("action");
        if (!action) EX_FAIL("PolicyBatchView::create: 缺少 action");
        auto action_mask = record.field("action_mask");
        if (!action_mask) EX_FAIL("PolicyBatchView::create: 缺少 action_mask");
        PolicyBatchView view(record);
        view.action_ = *action;
        view.action_mask_ = *action_mask;
        auto set_optional = [&](std::string_view name, std::optional<FieldHandle>& out)
        {
            auto h = record.optional_field(name);
            if (h && h->valid()) out = *h;
        };
        set_optional("old_log_prob", view.old_log_prob_);
        set_optional("advantage", view.advantage_);
        set_optional("return", view.returns_);
        set_optional("old_value", view.old_value_);
        set_optional("value_mask", view.value_mask_);
        set_optional("group_id", view.group_id_);
        return view;
    }

    [[nodiscard]] const TensorRecord& record() const noexcept { return *record_; }
    [[nodiscard]] const Tensor& action() const { return *record_->find(action_); }
    [[nodiscard]] const Tensor& action_mask() const { return *record_->find(action_mask_); }
    [[nodiscard]] const Tensor* old_log_prob() const noexcept
    { return old_log_prob_ ? record_->find(*old_log_prob_) : nullptr; }
    [[nodiscard]] const Tensor* advantage() const noexcept
    { return advantage_ ? record_->find(*advantage_) : nullptr; }
    [[nodiscard]] const Tensor* returns() const noexcept
    { return returns_ ? record_->find(*returns_) : nullptr; }
    [[nodiscard]] const Tensor* old_value() const noexcept
    { return old_value_ ? record_->find(*old_value_) : nullptr; }
    [[nodiscard]] const Tensor* value_mask() const noexcept
    { return value_mask_ ? record_->find(*value_mask_) : nullptr; }
    [[nodiscard]] const Tensor* group_id() const noexcept
    { return group_id_ ? record_->find(*group_id_) : nullptr; }
};

struct AdvantageConfig
{
    Scalar gamma = 0.99f;
    Scalar lambda = 0.95f;
    Precision precision = Precision::F32;
};

namespace detail
{
[[nodiscard]] inline exstl::Result<Tensor> rl_zero(ComputeEngine& engine, std::size_t rows,
                                            std::size_t cols, Precision p)
{
    EX_TRY(t, engine.try_create_tensor(rows, cols, p));
    EX_TRY_CHECK(engine.zero(*t));
    return t;
}

[[nodiscard]] inline exstl::Result<Tensor> rl_constant(ComputeEngine& engine, std::size_t rows,
                                                std::size_t cols, Scalar value, Precision p)
{
    EX_TRY(zero, rl_zero(engine, rows, cols, p));
    if (value == Scalar{0}) return zero;
    return dsl::compute(engine, dsl::leaf(*zero) + dsl::rparam(value), rows, cols, p);
}

inline exstl::Result<void> rl_check_same_shape(const Tensor& reference, const Tensor* other,
                                        const char* name)
{
    if (other == nullptr) return {};
    if (other->rows() != reference.rows() || other->cols() != reference.cols())
        EX_FAIL(std::string("RL transform: ") + name + " 形状与 reward/value 不一致（期望 " +
                reference.shape_str() + "，实际 " + other->shape_str() + "）");
    return {};
}
} // namespace detail

class GaeAdvantage
{
private:
    AdvantageConfig cfg_;

public:
    explicit GaeAdvantage(AdvantageConfig cfg = {}) : cfg_(cfg) {}

    struct Output
    {
        Tensor advantage;
        Tensor returns;
    };

    [[nodiscard]] exstl::Result<Output> apply(ComputeEngine& engine, const Tensor& rewards,
                                       const Tensor& values, const Tensor* done = nullptr,
                                       const Tensor* mask = nullptr,
                                       const Tensor* bootstrap_value = nullptr) const
    {
        const std::size_t steps = rewards.rows();
        const std::size_t batch = rewards.cols();
        if (steps == 0 || batch == 0) EX_FAIL("GaeAdvantage: reward 不能为空");
        if (values.rows() != steps || values.cols() != batch)
            EX_FAIL("GaeAdvantage: value 与 reward 形状不一致");
        EX_TRY_CHECK(detail::rl_check_same_shape(rewards, done, "done"));
        EX_TRY_CHECK(detail::rl_check_same_shape(rewards, mask, "mask"));
        if (bootstrap_value != nullptr &&
            (bootstrap_value->rows() != 1 || bootstrap_value->cols() != batch))
            EX_FAIL("GaeAdvantage: bootstrap_value 形状必须为 (1,batch)");

        const Precision p = cfg_.precision;
        EX_TRY(out_a, engine.try_create_tensor(steps, batch, p));
        EX_TRY(out_r, engine.try_create_tensor(steps, batch, p));
        EX_TRY(next_adv, detail::rl_zero(engine, 1, batch, p));
        Tensor next_value;
        if (bootstrap_value != nullptr) next_value = *bootstrap_value;
        else
        {
            EX_TRY(v, detail::rl_zero(engine, 1, batch, p));
            next_value = std::move(*v);
        }

        for (std::size_t ti = steps; ti-- > 0;)
        {
            EX_TRY(reward, engine.slice_rows(rewards, ti, 1));
            EX_TRY(value, engine.slice_rows(values, ti, 1));
            Tensor mask_row_owned;
            const Tensor* mask_row = nullptr;
            if (mask != nullptr)
            {
                EX_TRY(m, engine.slice_rows(*mask, ti, 1));
                mask_row_owned = std::move(*m);
                mask_row = &mask_row_owned;
            }

            Tensor continuation;
            if (done != nullptr)
            {
                EX_TRY(d, engine.slice_rows(*done, ti, 1));
                EX_TRY(c, dsl::compute(engine,
                                       dsl::leaf(*reward) * dsl::rparam(0) + dsl::rparam(1) -
                                           dsl::leaf(*d),
                                       1, batch, p));
                continuation = std::move(*c);
            }
            else
            {
                EX_TRY(c, detail::rl_constant(engine, 1, batch, 1, p));
                continuation = std::move(*c);
            }

            EX_TRY(delta, dsl::compute(engine,
                                       dsl::leaf(*reward) +
                                           dsl::leaf(next_value) * dsl::rparam(cfg_.gamma) *
                                               dsl::leaf(continuation) -
                                           dsl::leaf(*value),
                                       1, batch, p));
            EX_TRY(advance,
                   dsl::compute(engine,
                                dsl::leaf(*delta) + dsl::leaf(*next_adv) *
                                                        dsl::rparam(cfg_.gamma * cfg_.lambda) *
                                                        dsl::leaf(continuation),
                                1, batch, p));

            if (mask_row != nullptr)
            {
                EX_TRY(masked_a, dsl::compute(engine, dsl::leaf(*advance) * dsl::leaf(*mask_row), 1,
                                              batch, p));
                advance = std::move(*masked_a);
            }
            Tensor ret_row;
            if (mask_row != nullptr)
            {
                EX_TRY(ret,
                       dsl::compute(
                           engine, (dsl::leaf(*advance) + dsl::leaf(*value)) * dsl::leaf(*mask_row),
                           1, batch, p));
                ret_row = std::move(*ret);
            }
            else
            {
                EX_TRY(ret,
                       dsl::compute(engine, dsl::leaf(*advance) + dsl::leaf(*value), 1, batch, p));
                ret_row = std::move(*ret);
            }

            EX_TRY_CHECK(engine.insert_rows(*out_a, ti, *advance));
            EX_TRY_CHECK(engine.insert_rows(*out_r, ti, ret_row));
            next_adv = std::move(*advance);
            if (mask_row != nullptr)
            {
                EX_TRY(valid_value,
                       dsl::compute(engine, dsl::leaf(*value) * dsl::leaf(*mask_row), 1, batch, p));
                next_value = std::move(*valid_value);
            }
            else
            {
                next_value = *value;
            }
        }
        return Output{.advantage = std::move(*out_a), .returns = std::move(*out_r)};
    }
};

class MonteCarloReturn
{
private:
    Scalar gamma_ = 0.99f;
    Precision precision_ = Precision::F32;

public:
    explicit MonteCarloReturn(Scalar gamma = 0.99f, Precision precision = Precision::F32)
        : gamma_(gamma), precision_(precision)
    {
    }

    [[nodiscard]] exstl::Result<Tensor> apply(ComputeEngine& engine, const Tensor& rewards,
                                       const Tensor* done = nullptr,
                                       const Tensor* mask = nullptr) const
    {
        const std::size_t steps = rewards.rows();
        const std::size_t batch = rewards.cols();
        if (steps == 0 || batch == 0) EX_FAIL("MonteCarloReturn: reward 不能为空");
        EX_TRY_CHECK(detail::rl_check_same_shape(rewards, done, "done"));
        EX_TRY_CHECK(detail::rl_check_same_shape(rewards, mask, "mask"));

        EX_TRY(out, engine.try_create_tensor(steps, batch, precision_));
        EX_TRY(next, detail::rl_zero(engine, 1, batch, precision_));
        for (std::size_t ti = steps; ti-- > 0;)
        {
            EX_TRY(reward, engine.slice_rows(rewards, ti, 1));
            Tensor continuation;
            if (done != nullptr)
            {
                EX_TRY(d, engine.slice_rows(*done, ti, 1));
                EX_TRY(c, dsl::compute(engine,
                                       dsl::leaf(*reward) * dsl::rparam(0) + dsl::rparam(1) -
                                           dsl::leaf(*d),
                                       1, batch, precision_));
                continuation = std::move(*c);
            }
            else
            {
                EX_TRY(c, detail::rl_constant(engine, 1, batch, 1, precision_));
                continuation = std::move(*c);
            }
            EX_TRY(row, dsl::compute(engine,
                                     dsl::leaf(*reward) + dsl::leaf(*next) * dsl::rparam(gamma_) *
                                                              dsl::leaf(continuation),
                                     1, batch, precision_));
            if (mask != nullptr)
            {
                EX_TRY(m, engine.slice_rows(*mask, ti, 1));
                EX_TRY(masked,
                       dsl::compute(engine, dsl::leaf(*row) * dsl::leaf(*m), 1, batch, precision_));
                row = std::move(*masked);
            }
            EX_TRY_CHECK(engine.insert_rows(*out, ti, *row));
            next = std::move(*row);
        }
        return out;
    }
};

// 总体标准差（不是样本标准差）；mask=0 的项不参与均值/方差。
class MaskedNormalizer
{
private:
    Scalar epsilon_ = 1e-8f;
    Precision precision_ = Precision::F32;

public:
    explicit MaskedNormalizer(Scalar epsilon = 1e-8f, Precision precision = Precision::F32)
        : epsilon_(epsilon), precision_(precision)
    {
    }

    [[nodiscard]] exstl::Result<Tensor> apply(ComputeEngine& engine, const Tensor& values,
                                       const Tensor& mask) const
    {
        if (values.rows() == 0 || values.cols() == 0) EX_FAIL("MaskedNormalizer: values 不能为空");
        EX_TRY_CHECK(detail::rl_check_same_shape(values, &mask, "mask"));

        EX_TRY(weighted, dsl::compute(engine, dsl::leaf(values) * dsl::leaf(mask), values.rows(),
                                      values.cols(), precision_));
        EX_TRY(square, dsl::compute(engine, dsl::leaf(values) * dsl::leaf(values) * dsl::leaf(mask),
                                    values.rows(), values.cols(), precision_));
        EX_TRY(sum_w, dsl::compute_reduce(engine, dsl::col_reduce_sum(dsl::leaf(*weighted)),
                                          values.rows(), values.cols(), precision_));
        EX_TRY(sum_sq, dsl::compute_reduce(engine, dsl::col_reduce_sum(dsl::leaf(*square)),
                                           values.rows(), values.cols(), precision_));
        EX_TRY(sum_m, dsl::compute_reduce(engine, dsl::col_reduce_sum(dsl::leaf(mask)),
                                          values.rows(), values.cols(), precision_));
        EX_TRY(total_w, dsl::compute_reduce(engine, dsl::row_reduce_sum(dsl::leaf(*sum_w)), 1,
                                            values.cols(), precision_));
        EX_TRY(total_sq, dsl::compute_reduce(engine, dsl::row_reduce_sum(dsl::leaf(*sum_sq)), 1,
                                             values.cols(), precision_));
        EX_TRY(total_m, dsl::compute_reduce(engine, dsl::row_reduce_sum(dsl::leaf(*sum_m)), 1,
                                            values.cols(), precision_));
        auto w = nn::download_vector(engine, *total_w);
        EX_TRY_CHECK(w);
        auto sq = nn::download_vector(engine, *total_sq);
        EX_TRY_CHECK(sq);
        auto m = nn::download_vector(engine, *total_m);
        EX_TRY_CHECK(m);

        const Scalar count = (*m)[0];
        if (count <= Scalar{0})
            return detail::rl_zero(engine, values.rows(), values.cols(), precision_);
        const Scalar mean = (*w)[0] / count;
        Scalar variance = (*sq)[0] / count - mean * mean;
        if (variance < Scalar{0}) variance = Scalar{0};
        if (variance <= epsilon_ * epsilon_)
            return detail::rl_zero(engine, values.rows(), values.cols(), precision_);
        const Scalar inv_std = Scalar{1} / std::sqrt(variance);
        return dsl::compute(engine,
                            (dsl::leaf(values) - dsl::rparam(mean)) * dsl::rparam(inv_std) *
                                dsl::leaf(mask),
                            values.rows(), values.cols(), precision_);
    }
};

struct GroupSpan
{
    std::size_t first_row = 0;
    std::size_t row_count = 0;
};

class GroupAdvantage
{
private:
    Scalar epsilon_ = 1e-8f;
    Precision precision_ = Precision::F32;

public:
    explicit GroupAdvantage(Scalar epsilon = 1e-8f, Precision precision = Precision::F32)
        : epsilon_(epsilon), precision_(precision)
    {
    }

    // returns: (N,1)；spans 必须有序、互不重叠并连续覆盖全部 N 行。
    [[nodiscard]] exstl::Result<Tensor> apply(ComputeEngine& engine, const Tensor& returns,
                                       std::span<const GroupSpan> spans) const
    {
        if (returns.cols() != 1 || returns.rows() == 0)
            EX_FAIL("GroupAdvantage: returns 形状必须为 (N,1)");
        std::size_t cursor = 0;
        for (const GroupSpan& span : spans)
        {
            if (span.row_count == 0) EX_FAIL("GroupAdvantage: group row_count 必须 > 0");
            if (span.first_row != cursor) EX_FAIL("GroupAdvantage: group spans 必须连续且无重叠");
            cursor += span.row_count;
        }
        if (cursor != returns.rows()) EX_FAIL("GroupAdvantage: group spans 未覆盖全部 returns 行");

        EX_TRY(out, engine.try_create_tensor(returns.rows(), 1, precision_));
        MaskedNormalizer normalizer(epsilon_, precision_);
        for (const GroupSpan& span : spans)
        {
            EX_TRY(group, engine.slice_rows(returns, span.first_row, span.row_count));
            EX_TRY(mask, detail::rl_constant(engine, span.row_count, 1, 1, precision_));
            EX_TRY(normalized, normalizer.apply(engine, *group, *mask));
            EX_TRY_CHECK(engine.insert_rows(*out, span.first_row, *normalized));
        }
        return out;
    }
};

} // namespace nn
