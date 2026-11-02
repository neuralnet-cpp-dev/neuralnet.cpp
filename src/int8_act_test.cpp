// ── int8_act_test — int8 激活量化（W1.58A8，P3）─────────────────────────────
// 设计依据：docs/internal/21-quantized-weights.md §4.16（P3：激活 int8 +
// matmul_q_i8 整数点积；BitNet b1.58 训练配方"trained with 1.58-bit weights
// and 8-bit activations"的显存侧落地）。
//
// 覆盖：
//   [1] quantize_i8 原语：round/clamp 逐位符合参考（round-half-away、
//       s==0 → 0、NaN → 0）+ 打包布局（i8_pack::at 读回逐位一致）。
//   [2] unpack_i8：roundtrip 逐位（值 ∈ [-127,127] 整数）。
//   [3] matmul_q_i8：与宿主整数参考**逐位一致**（同累加序：k 升序 int32 +
//       (τ·s)·acc 缩放序）；跨形状（K 非 16 倍数 / N 非 4 倍数 / M 跨行块
//       边界；GPU 追加 2B4T 量级）。
//   [4] BitLinear int8-act forward vs 既有 f16 路径：差 ≤ 8-bit 量化界
//       （max_τ·K·max_s/254 —— s = absmax/127 的 W1.58A8 口径，docs 21
//       §4.16 ③；原值 absmax 会让激活量化退化成三值分辨率）。
//   [5] backward：dX 与 f16 路径逐位一致（同 matmul_q_t）；dW 的 STE 用
//       反量化激活 —— 与宿主参考容差一致。
//   [6] 端到端小训练收敛（CPU；`--gpu` 时 GPU）—— 全部 BitLinear 开 int8-act。
//
// 退出码：0 = 通过，1 = 失败，77 = 跳过（--gpu 且无设备）
// ─────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "neuralnet.cpp/cli/cli_engine_factory.hpp"
#include "neuralnet.cpp/compute_layer_bitlinear.hpp"
#include "neuralnet.cpp/int8_pack.hpp"
#include "neuralnet.cpp/ternary_pack.hpp"

#define NN_TEST_COUNTER g_failures
#include "test_common.hpp"

namespace
{

int g_failures = 0;

[[nodiscard]] nn::Matrix random_matrix(std::size_t rows, std::size_t cols, unsigned seed)
{
    nn::Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& v : m.span())
        v = d(rng);
    return m;
}

bool matrices_equal(const nn::Matrix& a, const nn::Matrix& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    const auto sa = a.span();
    const auto sb = b.span();
    for (std::size_t i = 0; i < sa.size(); ++i)
        if (sa[i] != sb[i]) return false;
    return true;
}

// ── [1] quantize_i8：round/clamp 逐位 + 打包布局 ──────────────────────────
void test_quantize(nn::ComputeEngine& e)
{
    std::puts("[1] quantize_i8：round/clamp 逐位（含 s==0 / NaN）+ 打包布局");
    constexpr std::size_t M = 5, N = 7;   // N 非 4 倍数（跨尾部字）
    nn::Matrix xm = random_matrix(M, N, 11);
    nn::Matrix sm(1, N);
    for (std::size_t c = 0; c < N; ++c)
    {
        float mx = 0.0f;
        for (std::size_t r = 0; r < M; ++r)
            mx = std::max(mx, std::fabs(xm.span()[r * N + c]));
        sm.span()[c] = mx;
    }
    // 注入边界：一整列 0（s==0）与一个 NaN
    for (std::size_t r = 0; r < M; ++r)
        xm.span()[r * N + 0] = 0.0f;
    xm.span()[1 * N + 1] = std::numeric_limits<float>::quiet_NaN();

    const nn::Tensor xt = upload(e, xm);
    const nn::Tensor st = upload(e, sm);
    auto q_r = e.quantize_i8(xt, st, /*col_scale=*/true);
    CHECK(q_r.has_value(), "quantize_i8 成功");
    if (!q_r.has_value()) return;
    const nn::Matrix qm = download(e, *q_r);
    const std::size_t wpr = nn::i8_pack::words_per_row(N);
    CHECK(qm.rows() == M && qm.cols() == wpr, "打包形状 (M, ceil(N/4))");

    bool layout_ok = true, value_ok = true;
    for (std::size_t r = 0; r < M; ++r)
    {
        std::vector<std::uint32_t> words(wpr);
        for (std::size_t wi = 0; wi < wpr; ++wi)
            words[wi] = std::bit_cast<std::uint32_t>(qm.span()[r * wpr + wi]);
        for (std::size_t c = 0; c < N; ++c)
        {
            const float s = sm.span()[c];
            const std::int8_t want = nn::i8_pack::quantize(xm.span()[r * N + c], s);
            const std::int8_t got = nn::i8_pack::at(words.data(), c);
            if (got != want) value_ok = false;
            // 布局：低位字节对应靠前元素
            if (c < N - 1)
            {
                const std::uint32_t w0 = words[c / 4], w1 = words[(c + 1) / 4];
                if ((w0 >> (8 * (c % 4)) & 0xFFu) != static_cast<std::uint8_t>(got))
                    layout_ok = false;
                (void)w1;
            }
        }
    }
    CHECK(value_ok, "量化值与宿主参考逐位一致（含 s==0 列 → 0、NaN → 0）");
    CHECK(layout_ok, "行内小端字节序");
}

// ── [2] unpack_i8：roundtrip 逐位 ─────────────────────────────────────────
void test_unpack(nn::ComputeEngine& e)
{
    std::puts("[2] unpack_i8：打包 → f32 整数 roundtrip 逐位");
    constexpr std::size_t M = 3, N = 6;   // N 非 4 倍数
    const nn::Matrix xm = random_matrix(M, N, 23);
    const nn::Tensor xt = upload(e, xm);
    nn::Matrix sm(1, N);
    for (std::size_t c = 0; c < N; ++c)
        sm.span()[c] = std::max(0.3f, std::fabs(xm.span()[c]) + 0.1f);
    const nn::Tensor st = upload(e, sm);
    auto q_r = e.quantize_i8(xt, st, true);
    CHECK(q_r.has_value(), "quantize_i8 成功");
    if (!q_r.has_value()) return;
    auto u_r = e.unpack_i8(*q_r, N);
    CHECK(u_r.has_value(), "unpack_i8 成功");
    if (!u_r.has_value()) return;
    const nn::Matrix um = download(e, *u_r);
    CHECK(um.rows() == M && um.cols() == N, "unpack 形状 (M,N)");
    // 值域 + 与量化值的逐位一致（整数升 f32 精确）
    bool ok = true;
    for (std::size_t i = 0; i < um.span().size(); ++i)
    {
        const float v = um.span()[i];
        if (v != static_cast<float>(static_cast<std::int8_t>(static_cast<int>(v))))
            ok = false;
        if (v < -127.0f || v > 127.0f) ok = false;
    }
    CHECK(ok, "unpack 值 ∈ [-127,127] 整数（f32 精确可表示）");
}

// ── [3] matmul_q_i8：与宿主整数参考逐位一致（跨形状）────────────────────
bool test_matmul_q_i8_shape(nn::ComputeEngine& e, std::size_t OUT, std::size_t K,
                            std::size_t N, unsigned seed, const char* label)
{
    std::printf("[3] matmul_q_i8（%s）：与宿主整数参考逐位一致\n", label);
    const nn::Matrix wm = random_matrix(OUT, K, seed);
    nn::Matrix tm(OUT, 1);
    for (std::size_t r = 0; r < OUT; ++r)
    {
        double acc = 0.0;
        for (std::size_t k = 0; k < K; ++k)
            acc += std::fabs(static_cast<double>(wm.span()[r * K + k]));
        tm.span()[r] = static_cast<float>(acc / static_cast<double>(K));
    }
    const nn::Matrix xm = random_matrix(K, N, seed + 1);
    nn::Matrix sm(1, N);
    for (std::size_t c = 0; c < N; ++c)
    {
        float mx = 0.0f;
        for (std::size_t k = 0; k < K; ++k)
            mx = std::max(mx, std::fabs(xm.span()[k * N + c]));
        sm.span()[c] = mx;
    }

    const nn::Tensor wt = upload(e, wm);
    const nn::Tensor tt = upload(e, tm);
    const nn::Tensor xt = upload(e, xm);
    const nn::Tensor st = upload(e, sm);
    auto pw_r = e.trit_pack(wt, tt);
    CHECK(pw_r.has_value(), "trit_pack 成功");
    if (!pw_r.has_value()) return false;
    auto px_r = e.quantize_i8(xt, st, /*col_scale=*/true);
    CHECK(px_r.has_value(), "quantize_i8 成功");
    if (!px_r.has_value()) return false;
    auto y_r = e.matmul_q_i8(*pw_r, tt, *px_r, st);
    CHECK(y_r.has_value(), "matmul_q_i8 成功");
    if (!y_r.has_value()) return false;
    const nn::Matrix ym = download(e, *y_r);
    const nn::Matrix qm = download(e, *px_r);

    // 宿主参考（与 CPU 实现同累加/缩放序：k 升序 int32；(τ·s)·float(acc)）
    const std::size_t wpr_w = nn::trit_pack::words_per_row(K);
    const std::size_t wpr_q = nn::i8_pack::words_per_row(N);
    std::vector<std::uint32_t> twords(wpr_w), qwords(wpr_q);
    nn::Matrix ref(OUT, N);
    for (std::size_t m = 0; m < OUT; ++m)
    {
        nn::trit_pack::pack_row(wm.span().data() + m * K, K, tm.span()[m], twords.data());
        for (std::size_t n = 0; n < N; ++n)
        {
            std::int32_t acc = 0;
            for (std::size_t k = 0; k < K; ++k)
            {
                const std::uint32_t wword = twords[k / nn::trit_pack::kPerWord];
                const std::uint32_t code =
                    (wword >> (nn::trit_pack::kBits * (k % nn::trit_pack::kPerWord))) &
                    nn::trit_pack::kMask;
                const std::int32_t trit = static_cast<std::int32_t>(nn::trit_pack::value_of(code));
                for (std::size_t wi = 0; wi < wpr_q; ++wi)
                    qwords[wi] = std::bit_cast<std::uint32_t>(qm.span()[k * wpr_q + wi]);
                const std::int32_t q = static_cast<std::int32_t>(nn::i8_pack::at(qwords.data(), n));
                acc += trit * q;
            }
            ref.span()[m * N + n] = tm.span()[m] * sm.span()[n] * static_cast<float>(acc);
        }
    }
    CHECK(matrices_equal(ym, ref), "逐位一致（k 升序 int32 + (τ·s)·acc 缩放序）");
    return true;
}

// ── [4] BitLinear int8-act forward vs f16 路径：差 ≤ 8-bit 量化界 ────────
bool test_bitlinear_forward(nn::ComputeEngine& e, const char* dev)
{
    std::printf("[4] BitLinear int8-act forward vs f16 路径（%s）：差 ≤ 量化界\n", dev);
    constexpr std::size_t IN = 64, OUT = 32, B = 8;
    nn::BitLinear layer(IN, OUT);
    nn::PrecisionProfile t1;
    t1.param = nn::Precision::T1_58;
    layer.set_precision_profile(t1);
    NN_UNWRAP(layer.init(e), "BitLinear init");
    const nn::Matrix xm = random_matrix(IN, B, 5);
    const nn::Tensor xt = upload(e, xm);
    auto y0 = layer.forward(xt);
    CHECK(y0.has_value(), "forward(f16 路径) 成功");
    if (!y0.has_value()) return false;
    layer.set_activation_quant(true);
    auto y1 = layer.forward(xt);
    CHECK(y1.has_value(), "forward(int8 路径) 成功");
    if (!y1.has_value()) return false;
    const nn::Matrix y0m = download(e, *y0);
    const nn::Matrix y1m = download(e, *y1);

    // 界：|Y1 − Y0| ≤ τ_m·s_n·K·0.5，其中 s_n = absmax_n/127（W1.58A8 口径）。
    // τ_m = per-row absmean(latent)（与层内 tau_ 同一口径）。
    nn::Matrix sm(1, B);
    for (std::size_t c = 0; c < B; ++c)
    {
        float mx = 0.0f;
        for (std::size_t r = 0; r < IN; ++r)
            mx = std::max(mx, std::fabs(xm.span()[r * B + c]));
        sm.span()[c] = mx / 127.0f;
    }
    const nn::Matrix lm = download(e, layer.parameters()[0].get());
    double worst = 0.0;
    for (std::size_t m = 0; m < OUT; ++m)
    {
        double tau = 0.0;
        for (std::size_t k = 0; k < IN; ++k)
            tau += std::fabs(static_cast<double>(lm.span()[m * IN + k]));
        tau /= static_cast<double>(IN);
        for (std::size_t n = 0; n < B; ++n)
        {
            const double bound = tau * static_cast<double>(sm.span()[n]) *
                                 static_cast<double>(IN) * 0.5;
            const double diff = std::fabs(static_cast<double>(y1m.span()[m * B + n]) -
                                          static_cast<double>(y0m.span()[m * B + n]));
            if (bound > 0.0) worst = std::max(worst, diff / bound);
        }
    }
    std::printf("    差/界 worst = %.4f\n", worst);
    CHECK(worst <= 1.0, "前向差 ≤ τ·K·(absmax/127)·0.5（8-bit 量化界）");
    return true;
}

// ── [5] backward：dX 逐位 + dW 与宿主参考一致 ────────────────────────────
bool test_bitlinear_backward(nn::ComputeEngine& e, const char* dev)
{
    std::printf("[5] BitLinear backward（%s）：dX 逐位 + dW 与宿主参考一致\n", dev);
    constexpr std::size_t IN = 6, OUT = 8, B = 3;
    nn::BitLinear layer(IN, OUT);
    nn::PrecisionProfile t1;
    t1.param = nn::Precision::T1_58;
    layer.set_precision_profile(t1);
    NN_UNWRAP(layer.init(e), "BitLinear init");
    const nn::Matrix xm = random_matrix(IN, B, 9);
    const nn::Matrix dm = random_matrix(OUT, B, 13);
    const nn::Tensor xt = upload(e, xm);
    const nn::Tensor dt = upload(e, dm);

    // 同一个层、同一份权重：先 f16 路径 forward/backward，清梯度后切 int8 路径。
    auto f0 = layer.forward(xt);
    CHECK(f0.has_value(), "forward(f16) 成功");
    if (!f0.has_value()) return false;
    auto gx0 = layer.backward(dt);
    CHECK(gx0.has_value(), "backward(f16) 成功");
    if (!gx0.has_value()) return false;
    NN_UNWRAP(layer.zero_grad(), "zero_grad");
    layer.set_activation_quant(true);
    auto f1 = layer.forward(xt);
    CHECK(f1.has_value(), "forward(int8) 成功");
    if (!f1.has_value()) return false;
    auto gx1 = layer.backward(dt);
    CHECK(gx1.has_value(), "backward(int8) 成功");
    if (!gx1.has_value()) return false;
    CHECK(matrices_equal(download(e, *gx0), download(e, *gx1)),
          "dX 逐位一致（两条路径同 matmul_q_t，act_quant 不泄漏进 dX）");

    // dW 宿主参考（int8 路径的 STE：用反量化激活）：s_n = absmax_n/127；
    // q = quantize(x, s)；dys[m,n] = d[m,n]·s_n；dW[m,k] = τ_m·Σ_n dys[m,n]·q(k,n)。
    const nn::Matrix gw = download(e, layer.param_gradients()[0].get());
    const nn::Matrix lm = download(e, layer.parameters()[0].get());
    nn::Matrix sm(1, B);
    for (std::size_t c = 0; c < B; ++c)
    {
        float mx = 0.0f;
        for (std::size_t r = 0; r < IN; ++r)
            mx = std::max(mx, std::fabs(xm.span()[r * B + c]));
        sm.span()[c] = mx / 127.0f;
    }
    nn::Matrix ref(OUT, IN);
    for (std::size_t m = 0; m < OUT; ++m)
    {
        double tau = 0.0;
        for (std::size_t k = 0; k < IN; ++k)
            tau += std::fabs(static_cast<double>(lm.span()[m * IN + k]));
        tau /= static_cast<double>(IN);
        for (std::size_t k = 0; k < IN; ++k)
        {
            double acc = 0.0;
            for (std::size_t n = 0; n < B; ++n)
            {
                const float s = sm.span()[n];
                const std::int8_t q = nn::i8_pack::quantize(xm.span()[k * B + n], s);
                acc += static_cast<double>(dm.span()[m * B + n] * s) * static_cast<double>(q);
            }
            ref.span()[m * IN + k] = static_cast<float>(tau * acc);
        }
    }
    const bool dw_ok = close_to(gw, ref, 1e-3f, "dW(int8-act) vs 宿主参考", 0);
    CHECK(dw_ok, "dW 与宿主参考一致（引擎 DSL 累加序 ≠ 宿主，容差 1e-3）");

    const nn::Matrix gb = download(e, layer.param_gradients()[1].get());
    CHECK(gb.rows() == OUT && gb.cols() == 1, "db 形状 (OUT,1)");
    return true;
}

// ── [6] 端到端小训练收敛（全部 BitLinear 开 int8-act）────────────────────
bool test_training(nn::ComputeEngine& e, const char* dev)
{
    std::printf("[6] 端到端小训练（%s）：全部 BitLinear 开 int8-act\n", dev);
    constexpr std::size_t IN = 8, HID = 16, OUT = 4, B = 32;

    // 教师：三值权重线性映射（学生是"BitLinear → Norm → 激活 → BitLinear"）
    std::mt19937 rng(20261010u);
    std::uniform_int_distribution<int> td(0, 2);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    nn::Matrix Wt(OUT, IN);
    for (auto& v : Wt.span())
        v = static_cast<float>(td(rng) - 1) * 0.5f;
    nn::Matrix Xm(IN, B);
    for (auto& v : Xm.span())
        v = ud(rng);
    nn::Matrix Ym(OUT, B);
    Wt.multiply_to(Ym, Xm);

    nn::PrecisionProfile t1;
    t1.param = nn::Precision::T1_58;
    auto m_r = nn::build_mnist_mlp_model(e, {IN, HID, OUT}, nn::NormType::LayerNorm, t1);
    CHECK(m_r.has_value(), "T1_58 MLP 构建成功");
    if (!m_r.has_value())
    {
        std::printf("    构建失败: %s\n", m_r.error().message.c_str());
        return false;
    }
    nn::Model& model = *m_r;
    std::size_t n_quant = 0;
    for (std::size_t li = 0; li < model.num_layers(); ++li)
        if (auto* bl = dynamic_cast<nn::BitLinear*>(&model.layer_at(li)))
        {
            bl->set_activation_quant(true);
            ++n_quant;
        }
    CHECK(n_quant >= 2, "至少两个 BitLinear 层开 int8-act");

    auto opt = nn::create_optimizer("adam", e, model.parameters(), model.param_gradients(),
                                    /*lr=*/0.05f, /*wd=*/0.0f, t1);
    CHECK(opt != nullptr, "优化器创建成功");
    if (!opt) return false;

    nn::MSELoss mse;
    mse.set_precision_profile(t1);
    const nn::Tensor Xt = upload(e, Xm);
    const nn::Tensor Yt = upload(e, Ym);

    float first = -1.f, last = -1.f;
    constexpr int STEPS = 300;
    for (int step = 0; step < STEPS; ++step)
    {
        if (auto r = model.zero_grad(); !r)
        {
            CHECK(false, "zero_grad");
            return false;
        }
        auto pred = model.forward(Xt);
        if (!pred)
        {
            CHECK(false, "forward");
            return false;
        }
        auto loss = mse.forward(e, *pred, Yt);
        if (!loss)
        {
            CHECK(false, "loss");
            return false;
        }
        if (!std::isfinite(*loss))
        {
            CHECK(false, "loss 非有限");
            return false;
        }
        if (step == 0) first = *loss;
        last = *loss;
        auto grad = mse.backward();
        if (!grad)
        {
            CHECK(false, "loss.backward");
            return false;
        }
        if (auto r = model.backward(*grad); !r)
        {
            CHECK(false, "model.backward");
            return false;
        }
        if (auto r = opt->step(); !r)
        {
            CHECK(false, "optimizer.step");
            return false;
        }
    }
    std::printf("    loss: %.6f → %.6f（%d 步）\n", first, last, STEPS);
    CHECK(last < first * 0.7f, "loss 显著下降（int8-act + 三值权重仍可训练）");
    return true;
}

}   // namespace

int main(int argc, char* argv[])
{
    bool gpu = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gpu") gpu = true;

    const auto run_all = [&](nn::ComputeEngine& e, const char* dev) {
        test_quantize(e);
        test_unpack(e);
        test_matmul_q_i8_shape(e, 130, 17, 6, 31, "K/N 非倍数 + 跨行块");
        if (gpu) test_matmul_q_i8_shape(e, 512, 256, 8, 41, "2B4T 量级");
        test_bitlinear_forward(e, dev);
        test_bitlinear_backward(e, dev);
        test_training(e, dev);
    };

    if (gpu)
    {
        nn::cli::EngineConfig ec;
        ec.use_gpu = true;
        auto eng_r = nn::cli::create_engine(ec);
        NN_EXIT(eng_r, 77, "GPU 不可用，跳过: ");
        run_all(**eng_r, "GPU");
    }
    else
    {
        auto eng_owner = nn::create_cpu_engine();
        run_all(*eng_owner, "CPU");
    }

    if (g_failures == 0) std::puts("int8_act_test: 全部通过");
    return g_failures == 0 ? 0 : 1;
}
