// ───────────────────────────────────────────────────────────────────────────
//  expr_fold_test.cpp — 分块状态归约（FoldSpec）CPU 语义对拍
//
//  覆盖（通用样例走共享构造 expr_fold.hpp、注意力样例走
//  compute_layer_attention.hpp —— 结构与 JIT 现编同源（AOT 收集链已删，P3），
//  保证 key/结构一致）：
//    1. rowmax fold        vs 独立逐行 max 参考
//    2. rowsum fold        vs 独立逐行 sum 参考
//    3. softmax_denom fold（online 双状态：m 跨块更新 + l = l·α + Σexp(x−m')）
//                          vs 独立 Σ_k exp(x − max_k x) 参考
//    4. K 边界族：{1, 7, 32, 33, 100, 1000}——单列 / 非块整除 / 整除 / 尾块
//    5. attention fold：5 掩码变体（含 Doc/AlibiDoc）× 4 形状族 vs 独立公式
//       参考；{2,133,4} 跨 EXPR_FOLD_BLOCK=128 → 多块流式 + causal 整块跳过
//    6. validate 负例：状态吃元素源 / fold+顶层 instrs / fold+matmul /
//       finalize 读输入 —— 四类违规必须被拒绝（静态拒而非静默错算）
//    7. 注册表去重 + key 契约（原 bin 序列化 roundtrip 的接替者：bin 已删除）：
//       同结构重复 add 只留一条；tri_skip（只翻该字段）与掩码 body 差异均进 key
//    8. 锚点代表值契约：注意力 fold 自登记（P4-4 ①）只登记一组代表形状，故
//       同一组合跨形状 key 必须相同；10 个组合（掩码 × 偏置 × {MHA,GQA}）互不相同
//  纯 CPU；输出 (rows,1)（eval_expr 的 cols 参数 = 1 调用约定）。
// ───────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "test_common.hpp" // M1：upload/download（库外经引擎进出张量）
import nn.engine;
import nn.expr;
import nn.layer;
using nn::Scalar;


int test_expr_fold()
{
    int fail = 0;
    const auto check = [&fail](bool ok, const std::string& msg)
    {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << msg << "\n";
        if (!ok) ++fail;
    };
    // NaN 守卫：diff 为 NaN/Inf（被测端输出 NaN）时误差记为 inf → 必超容差。
    //   IEEE fmax(err, NaN) = err 会静默吞掉 NaN diff——若无此守卫，被测端
    //   输出 NaN 时对拍仍可能 PASS，故此处直接把非有限 diff 记为 inf。
    const auto err_of = [](Scalar err, Scalar diff) -> Scalar
    {
        return std::isfinite(diff) ? std::fmax(err, diff) : std::numeric_limits<Scalar>::infinity();
    };

    auto engine_owner = nn::create_cpu_engine();
    nn::ComputeEngine& engine = *engine_owner;
    std::mt19937 rng(20260923);
    std::uniform_real_distribution<Scalar> dist(-2.0f, 2.0f);
    constexpr Scalar kTol = Scalar{1e-4};

    for (const std::size_t K : {std::size_t{1}, std::size_t{7}, std::size_t{32}, std::size_t{33},
                                std::size_t{100}, std::size_t{1000}})
    {
        const std::size_t rows = 5;
        nn::Matrix mx(rows, K);
        {
            auto sp = mx.span();
            for (auto& v : sp)
                v = dist(rng);
        }
        nn::Tensor x = upload(engine, mx);
        const std::array<nn::Tensor, 1> ins{x};

        // ── 1) rowmax ──
        {
            nn::ExprSpec spec = nn::expr::make_fold_rowmax(static_cast<std::uint32_t>(K));
            check(nn::validate_expr_spec(spec, 1).has_value(),
                  "rowmax validate K=" + std::to_string(K));
            auto out = engine.eval_expr(spec, ins, rows, 1);
            if (!out)
            {
                check(false, "rowmax eval K=" + std::to_string(K) + ": " + out.error().message);
            }
            else
            {
                const auto om = download(engine, *out);
                const auto xm = download(engine, x);
                const auto os = om.span();
                const auto xs = xm.span();
                Scalar err = 0;
                for (std::size_t r = 0; r < rows; ++r)
                {
                    Scalar ref = -std::numeric_limits<Scalar>::infinity();
                    for (std::size_t k = 0; k < K; ++k)
                        ref = std::fmax(ref, xs[r * K + k]);
                    err = err_of(err, std::fabs(os[r] - ref));
                }
                check(err <= kTol,
                      "rowmax vs ref K=" + std::to_string(K) + " err=" + std::to_string(err));
            }
        }

        // ── 2) rowsum ──
        {
            nn::ExprSpec spec = nn::expr::make_fold_rowsum(static_cast<std::uint32_t>(K));
            check(nn::validate_expr_spec(spec, 1).has_value(),
                  "rowsum validate K=" + std::to_string(K));
            auto out = engine.eval_expr(spec, ins, rows, 1);
            if (!out)
            {
                check(false, "rowsum eval K=" + std::to_string(K) + ": " + out.error().message);
            }
            else
            {
                const auto om = download(engine, *out);
                const auto xm = download(engine, x);
                const auto os = om.span();
                const auto xs = xm.span();
                Scalar err = 0;
                for (std::size_t r = 0; r < rows; ++r)
                {
                    Scalar ref = 0;
                    for (std::size_t k = 0; k < K; ++k)
                        ref += xs[r * K + k];
                    err =
                        err_of(err, std::fabs(os[r] - ref) / std::fmax(Scalar{1}, std::fabs(ref)));
                }
                check(err <= kTol,
                      "rowsum vs ref K=" + std::to_string(K) + " err=" + std::to_string(err));
            }
        }

        // ── 3) softmax 分母（online 双状态）──
        {
            nn::ExprSpec spec = nn::expr::make_fold_softmax_denom(static_cast<std::uint32_t>(K));
            check(nn::validate_expr_spec(spec, 1).has_value(),
                  "softmax_denom validate K=" + std::to_string(K));
            auto out = engine.eval_expr(spec, ins, rows, 1);
            if (!out)
            {
                check(false,
                      "softmax_denom eval K=" + std::to_string(K) + ": " + out.error().message);
            }
            else
            {
                const auto om = download(engine, *out);
                const auto xm = download(engine, x);
                const auto os = om.span();
                const auto xs = xm.span();
                Scalar err = 0;
                for (std::size_t r = 0; r < rows; ++r)
                {
                    Scalar m = -std::numeric_limits<Scalar>::infinity();
                    for (std::size_t k = 0; k < K; ++k)
                        m = std::fmax(m, xs[r * K + k]);
                    Scalar ref = 0;
                    for (std::size_t k = 0; k < K; ++k)
                        ref += std::exp(xs[r * K + k] - m);
                    err =
                        err_of(err, std::fabs(os[r] - ref) / std::fmax(Scalar{1}, std::fabs(ref)));
                }
                check(err <= kTol, "softmax_denom vs ref K=" + std::to_string(K) +
                                       " err=" + std::to_string(err));
            }
        }
    }

    // ── 6) validate 负例（静态拒绝）──
    {
        const auto rejects = [&](nn::ExprSpec spec, const char* msg)
        {
            const bool ok = !nn::validate_expr_spec(spec, spec.views.size()).has_value();
            check(ok, std::string("validate rejects ") + msg);
        };
        // (a) 状态吃元素类源（Input → 状态，kb 依赖覆盖）
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            s.fold->body = {};
            nn::ExprInstr b;
            b.op = static_cast<uint8_t>(nn::ExprOp::Add);
            b.dst = 0;
            b.a = nn::expr::input(0);
            b.b = nn::expr::cst(0);
            s.fold->body = {b};
            rejects(s, "state written from element source");
        }
        // (b) fold + 顶层 instrs 非空
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            nn::ExprInstr t;
            t.op = static_cast<uint8_t>(nn::ExprOp::Add);
            t.dst = 1;
            t.a = nn::expr::reg(0);
            t.b = nn::expr::cst(0);
            s.instrs = {t};
            rejects(s, "fold + non-empty top-level instrs");
        }
        // (c) fold + matmul 段互斥
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            nn::MatmulSpec mm;
            mm.a_input = 0;
            mm.b_input = 0;
            mm.k = 16;
            mm.batch = 1;
            s.matmul = mm;
            rejects(s, "fold + matmul segment");
        }
        // (d) finalize 读输入
        {
            nn::ExprSpec s = nn::expr::make_fold_rowsum(16);
            s.fold->finalize = {};
            nn::ExprInstr t;
            t.op = static_cast<uint8_t>(nn::ExprOp::Add);
            t.dst = 0;
            t.a = nn::expr::reg(0);
            t.b = nn::expr::input(0);
            s.fold->finalize = {t};
            rejects(s, "finalize reading inputs");
        }
    }

    // ── attention fold 对拍（独立公式参考；5 掩码 × 4 形状族）────────
    // 确定性迷你 case：seq=dk=bh=1，Q=2 K=3 Vt=5 → s=6, softmax=1, O=5
    {
        nn::Matrix mQ(1, 1), mK(1, 1), mVt(1, 1);
        mQ.span()[0] = 2;
        mK.span()[0] = 3;
        mVt.span()[0] = 5;
        nn::Tensor Q = upload(engine, mQ);
        nn::Tensor K = upload(engine, mK);
        nn::Tensor Vt = upload(engine, mVt);
        nn::ExprSpec sp = nn::expr::make_fold_attn_o(1, 1, 1, nn::expr::AttnMaskKind::Plain,
                                                     /*score_bias=*/false);
        const std::vector<nn::Tensor> ii{Q, K, Vt};
        auto o = engine.eval_expr(sp, ii, 1, 1);
        Scalar got = Scalar{-999};
        if (o)
        {
            const auto om = download(engine, *o);
            got = om.span()[0];
        }
        check(o.has_value() && std::fabs(got - Scalar{5}) < Scalar{1e-5},
              "attn fold mini(1x1) O=" + std::to_string(got) + " expect 5" +
                  (o ? "" : (" err=" + o.error().message)));
        // 分辨"多 j"vs"多行"：
        //   A 单行多 j：bh=1 seq=4 dk=1；B 多行单 j：bh=2 seq=1 dk=1
        for (int ab = 0; ab < 2; ++ab)
        {
            const std::uint32_t bh2 = ab == 0 ? 1u : 2u;
            const std::uint32_t seq2 = ab == 0 ? 4u : 1u;
            nn::Matrix mQ2(static_cast<std::size_t>(bh2) * 1, seq2);
            nn::Matrix mK2(static_cast<std::size_t>(bh2) * 1, seq2);
            nn::Matrix mV2(static_cast<std::size_t>(bh2) * seq2, 1);
            for (std::size_t i = 0; i < mQ2.span().size(); ++i)
                mQ2.span()[i] = Scalar{2};
            for (std::size_t i = 0; i < mK2.span().size(); ++i)
                mK2.span()[i] = Scalar{3};
            for (std::size_t i = 0; i < mV2.span().size(); ++i)
                mV2.span()[i] = Scalar{5};
            nn::Tensor Q2 = upload(engine, mQ2);
            nn::Tensor K2 = upload(engine, mK2);
            nn::Tensor V2 = upload(engine, mV2);
            nn::ExprSpec sp2 = nn::expr::make_fold_attn_o(
                seq2, 1, bh2, nn::expr::AttnMaskKind::Plain, /*score_bias=*/false);
            const std::vector<nn::Tensor> i2{Q2, K2, V2};
            auto o2 = engine.eval_expr(sp2, i2, bh2 * seq2, 1);
            std::string vals = "?";
            if (o2)
            {
                vals.clear();
                const auto om2 = download(engine, *o2);
                const auto sp2o = om2.span();
                for (std::size_t i = 0; i < bh2 * seq2; ++i)
                    vals += " " + std::to_string(sp2o[i]);
            }
            // 常数 Q=2,K=3 → 每行 softmax=1 → O = V = 5（所有行）
            check(o2.has_value(), std::string("mini-") + (ab ? "B" : "A") + " eval" +
                                      (o2 ? "" : (": " + o2.error().message)));
            if (o2)
            {
                bool okv = true;
                const auto om2 = download(engine, *o2);
                const auto sp2o = om2.span();
                for (std::size_t i = 0; i < bh2 * seq2; ++i)
                    if (std::fabs(sp2o[i] - Scalar{5}) > Scalar{1e-5}) okv = false;
                check(okv, std::string("mini-") + (ab ? "B(bh2seq1)" : "A(seq4)") + " O=" + vals +
                               " expect 5×" + std::to_string(bh2 * seq2));
            }
        }
        // C: 非均匀 Vt、Q=K=0（s 全 0 → p=1/4 均匀）→ 手算期望：
        //   d0 = 0.25·(10+20+30+40) = 25；d1 = 0.25·(1+2+3+4) = 2.5
        {
            nn::Matrix mQ3(2, 4), mK3(2, 4), mV3(4, 2); // Q/K: (bh·dk, seq)=(2,4)
            for (auto& v : mQ3.span())
                v = 0;
            for (auto& v : mK3.span())
                v = 0;
            const Scalar vv[8] = {10, 1, 20, 2, 30, 3, 40, 4};
            for (int i = 0; i < 8; ++i)
                mV3.span()[i] = vv[i];
            nn::Tensor Q3 = upload(engine, mQ3);
            nn::Tensor K3 = upload(engine, mK3);
            nn::Tensor V3 = upload(engine, mV3);
            nn::ExprSpec sp3 = nn::expr::make_fold_attn_o(4, 2, 1, nn::expr::AttnMaskKind::Plain,
                                                          /*score_bias=*/false);
            const std::vector<nn::Tensor> i3{Q3, K3, V3};
            auto o3 = engine.eval_expr(sp3, i3, 4, 2);
            check(o3.has_value(), "mini-C eval" + (o3 ? "" : (": " + o3.error().message)));
            if (o3)
            {
                const auto om3 = download(engine, *o3);
                const auto s3 = om3.span();
                std::string vals;
                for (std::size_t i = 0; i < 8; ++i)
                    vals += " " + std::to_string(s3[i]);
                // Q=K=0 → S 全 0 → p 均匀 1/4：全部 4 行输出相同
                //   d0 = (10+20+30+40)/4 = 25；d1 = (1+2+3+4)/4 = 2.5（全量比对）
                bool okc = true;
                for (std::size_t i = 0; i < 8; ++i)
                {
                    const Scalar want = (i % 2 == 0) ? Scalar{25} : Scalar{2.5};
                    if (std::fabs(s3[i] - want) >= Scalar{1e-4}) okc = false;
                }
                check(okc, "mini-C O=" + vals + " expect 4×(25 2.5)");
            }
        }
    }

    struct Mk
    {
        nn::expr::AttnMaskKind mask;
        bool score_bias;
        const char* name;
    };
    for (const Mk mk : {Mk{nn::expr::AttnMaskKind::Plain, false, "plain"},
                        Mk{nn::expr::AttnMaskKind::Causal, false, "causal"},
                        Mk{nn::expr::AttnMaskKind::Causal, true, "alibi"},
                        Mk{nn::expr::AttnMaskKind::CausalDoc, false, "doc"},
                        Mk{nn::expr::AttnMaskKind::CausalDoc, true, "alibidoc"}})
    {
        const char* mname = mk.name;
        struct Sh
        { std::uint32_t bh, seq, dk; };
        // {2,133,4}：seq > EXPR_FOLD_BLOCK(128) → 跨块流式（m/l 进位 + vecacc
        //   rescale）与 causal 整块跳过分支（k0>qt 空块）首次被执行；
        //   其余形状覆盖单块与奇数行（clamp + row_ok）
        for (const Sh sh : {Sh{2, 9, 4}, Sh{1, 33, 8}, Sh{3, 5, 2}, Sh{2, 133, 4}})
        {
            const std::size_t rows_out = static_cast<std::size_t>(sh.bh) * sh.seq;
            nn::Matrix mQ(static_cast<std::size_t>(sh.bh) * sh.dk, sh.seq);
            nn::Matrix mK(static_cast<std::size_t>(sh.bh) * sh.dk, sh.seq);
            nn::Matrix mVt(rows_out, sh.dk);
            nn::Matrix mslopes(1, sh.bh);
            // doc 输入（与 Layer 组包同序：Q,K,Vt,[slopes],doc_col,doc_ids）：
            //   doc_col (rows,1) = 行（查询位置）文档 id；doc_ids (1, bh*seq)
            //   按 (b,h) 块重复——BatchCol(seq) 读 [块*seq+j]，heads>1 时
            //   (1, batch*seq) 会越界（AGENTS 教训 #2）
            const std::uint32_t sseq = sh.seq;
            nn::Matrix mdoc_col(rows_out, 1);
            nn::Matrix mdoc_ids_t(1, static_cast<std::size_t>(sh.bh) * sh.seq);
            // doc 分段边界：seq > EXPR_FOLD_BLOCK(128) 时放在 **128 之后**
            //   （如 133 → 129），使查询位置 i ≥ 边界的行其首个 fold 块
            //   (j<128) 被文档掩码**全部**屏蔽——触发首块全屏蔽的分支
            //   （max init=-inf 时 m_old=blk_m=-inf → dm=−inf−−inf=NaN）。
            //   边界若落在 128 之内（如 seq/2），首块永远留有有效项，
            //   该分支测不到。seq ≤ 128 时用 seq/2 二分段。
            const std::uint32_t doc_boundary =
                sseq > nn::EXPR_FOLD_BLOCK ? nn::EXPR_FOLD_BLOCK + 1u : sseq / 2u;
            const auto doc_of = [doc_boundary](std::uint32_t pos) -> Scalar
            { return pos < doc_boundary ? Scalar{1} : Scalar{2}; };
            for (std::size_t r = 0; r < rows_out; ++r)
                mdoc_col.span()[r] = doc_of(static_cast<std::uint32_t>(r % sh.seq));
            for (std::uint32_t blk = 0; blk < sh.bh; ++blk)
                for (std::uint32_t j = 0; j < sh.seq; ++j)
                    mdoc_ids_t.span()[static_cast<std::size_t>(blk) * sh.seq + j] = doc_of(j);
            for (auto& v : mQ.span())
                v = dist(rng);
            for (auto& v : mK.span())
                v = dist(rng);
            for (auto& v : mVt.span())
                v = dist(rng);
            for (auto& v : mslopes.span())
                v = dist(rng) * Scalar{0.1};
            // 宿主矩阵全部写完后经引擎上传（出生绑定同一 engine）
            nn::Tensor Q = upload(engine, mQ);
            nn::Tensor K = upload(engine, mK);
            nn::Tensor Vt = upload(engine, mVt);
            nn::Tensor slopes = upload(engine, mslopes);
            nn::Tensor doc_col = upload(engine, mdoc_col);
            nn::Tensor doc_ids_t = upload(engine, mdoc_ids_t);
            const bool alibi = mk.score_bias;
            const bool docm = (mk.mask == nn::expr::AttnMaskKind::CausalDoc);
            std::vector<nn::Tensor> ins{Q, K, Vt};
            if (alibi) ins.push_back(slopes);
            if (docm)
            {
                ins.push_back(doc_col);
                ins.push_back(doc_ids_t);
            }

            nn::ExprSpec spec =
                nn::expr::make_fold_attn_o(sh.seq, sh.dk, sh.bh, mk.mask, mk.score_bias);
            if (auto v = nn::validate_expr_spec(spec, ins.size()); !v)
            {
                check(false, std::string("attn fold ") + mname + " validate: " + v.error().message);
                continue;
            }
            auto out = engine.eval_expr(spec, ins, rows_out, sh.dk);
            if (!out)
            {
                check(false, std::string("attn fold ") + mname + " eval: " + out.error().message);
                continue;
            }

            // 独立公式参考：per (b, i)：s_j = Σ_d Q[(b·dk+d),i]·K[(b·dk+d),j]
            //   [+掩码] → m → p=exp(s−m) → l → O[i,d] = Σ_j p_j·Vt[j,d] / l
            const auto Qm = download(engine, Q);
            const auto Km = download(engine, K);
            const auto Vtm = download(engine, Vt);
            const auto Slm = download(engine, slopes);
            const auto om = download(engine, *out);
            const auto qs = Qm.span();
            const auto ks = Km.span();
            const auto vs = Vtm.span();
            const auto sl = Slm.span();
            const auto os = om.span();
            Scalar err = 0;
            std::vector<Scalar> srow(sh.seq), prow(sh.seq);
            std::vector<Scalar> ref_out(static_cast<std::size_t>(rows_out) * sh.dk, 0);
            for (std::uint32_t b = 0; b < sh.bh; ++b)
            {
                for (std::uint32_t i = 0; i < sh.seq; ++i)
                {
                    Scalar m = -std::numeric_limits<Scalar>::infinity();
                    for (std::uint32_t j = 0; j < sh.seq; ++j)
                    {
                        Scalar acc = 0;
                        for (std::uint32_t d = 0; d < sh.dk; ++d)
                            acc +=
                                qs[(b * sh.dk + d) * sh.seq + i] * ks[(b * sh.dk + d) * sh.seq + j];
                        if (mk.mask != nn::expr::AttnMaskKind::Plain && j > i)
                            acc = -std::numeric_limits<Scalar>::infinity();
                        // 文档块对角：跨文档 -inf（fold body 内 causal→doc→alibi
                        //   链序；-inf 加有限斜率项不改值）——边界与 doc_of 同源
                        if (docm && ((i < doc_boundary) != (j < doc_boundary)))
                            acc = -std::numeric_limits<Scalar>::infinity();
                        if (alibi)
                            acc += sl[b] *
                                   static_cast<Scalar>(static_cast<int>(j) - static_cast<int>(i));
                        srow[j] = acc;
                        m = std::fmax(m, acc);
                    }
                    Scalar l = 0;
                    for (std::uint32_t j = 0; j < sh.seq; ++j)
                    {
                        prow[j] = std::exp(srow[j] - m);
                        l += prow[j];
                    }
                    for (std::uint32_t d = 0; d < sh.dk; ++d)
                    {
                        Scalar acc = 0;
                        for (std::uint32_t j = 0; j < sh.seq; ++j)
                            acc += prow[j] * vs[(b * sh.seq + j) * sh.dk + d];
                        const Scalar ref = acc / l;
                        const std::size_t oi = static_cast<std::size_t>(b) * sh.seq + i;
                        ref_out[oi * sh.dk + d] = ref;
                        err = err_of(err, std::fabs(os[oi * sh.dk + d] - ref) /
                                              std::fmax(Scalar{1}, std::fabs(ref)));
                    }
                }
            }
            check(err <= kTol, std::string("attn fold ") + mname + " bh=" + std::to_string(sh.bh) +
                                   " seq=" + std::to_string(sh.seq) +
                                   " dk=" + std::to_string(sh.dk) + " err=" + std::to_string(err));
            if (err > kTol)
            {
                std::cout << "      out[0..5]:";
                for (std::size_t i = 0; i < 6 && i < os.size(); ++i)
                    std::cout << " " << os[i];
                std::cout << "\n      ref[0..5]:";
                for (std::size_t i = 0; i < 6 && i < ref_out.size(); ++i)
                    std::cout << " " << ref_out[i];
                std::cout << "\n";
            }
        }
    }

    // ── 7) 注册表去重 + key 契约（原 bin 序列化 roundtrip 的接替者）────────
    // 原先这里验证 write_registry/read_registry 往返不丢 tri_skip 位。bin 序列化
    // 与其后的 AOT 收集链均已删除（P3，35 号 D4），改验结构 key 的两条不变量：
    // ① 同结构重复构造 key 唯一；② tri_skip 进 key。
    {
        nn::fused::ExprRegistry reg;
        const nn::ExprSpec sk = nn::expr::make_fold_attn_o(64, 4, 2, nn::expr::AttnMaskKind::Causal,
                                                           /*score_bias=*/false); // tri_skip=true
        const nn::ExprSpec pl = nn::expr::make_fold_attn_o(64, 4, 2, nn::expr::AttnMaskKind::Plain,
                                                           /*score_bias=*/false); // tri_skip=false
        reg.add(sk);
        reg.add(pl);
        reg.add(nn::expr::make_fold_rowsum(64));
        const std::size_t n1 = reg.specs.size();
        reg.add(sk); // 重复 add → 按 key 去重
        check(n1 == 3 && reg.specs.size() == 3,
              "registry add 去重 n=" + std::to_string(reg.specs.size()));
        // tri_skip 进 key 必须**只翻该字段**来验：Causal vs Plain 的 body 本就
        // 不同（has_causal 多 3 条 Gt/Select/Add 指令，指令序列进 key），拿它们
        // 比 key 即使删掉 `feed(&tri_skip, 1)` 也照样"通过" = 假测试。
        {
            nn::ExprSpec a = sk;
            nn::ExprSpec b = sk;
            b.fold->tri_skip = !a.fold->tri_skip;
            check(nn::expr_spec_key(nn::canonicalize_expr_spec(a)) !=
                      nn::expr_spec_key(nn::canonicalize_expr_spec(b)),
                  "fold tri_skip 进 key：同结构仅翻 tri_skip 位后 key 必须不同");
        }
        // 顺带锁死"不同掩码 = 不同结构"（body 差异进 key）
        check(nn::expr_spec_key(nn::canonicalize_expr_spec(sk)) !=
                  nn::expr_spec_key(nn::canonicalize_expr_spec(pl)),
              "Causal vs Plain 掩码 body 不同 → key 不同");
        bool saw_skip = false;
        for (const auto& s : reg.specs)
            if (s.fold && s.fold->tri_skip) saw_skip = true;
        check(saw_skip, "fold Causal → tri_skip=true 登记保留");
    }

    // ── 8) 锚点代表值契约（P4-4 ①）───────────────────────────────────────
    // 注意力 fold 的结构自登记（`compute_layer_attention.hpp` 的
    // `attn_fold_anchor`）只登记**一组代表形状**（seq=64/dk=2/bh=2，kv_ratio
    // 取 1 与 2）——它能覆盖所有真实调用点的前提 = "这些量都不进 key"。本段把
    // 该前提锁成契约（违反 → 注册表对真实形状 miss，走 JIT 兜底）：
    //   ① 同一组合在任意形状下 key 相同（含 GQA 比：2 ≡ 4，任意 n_rep 共用）；
    //   ② 10 个组合（掩码清单 × {无偏置, 位置偏置} −"Plain+偏置"，× {MHA,GQA}）
    //      两两不同（登记少一条 = GPU 硬报错）。
    {
        const auto key_of = [](nn::expr::AttnMaskKind m, bool sb, std::uint32_t kv_ratio,
                               std::uint32_t seq, std::uint32_t dk, std::uint32_t bh)
        {
            return nn::expr_spec_key(nn::canonicalize_expr_spec(
                nn::expr::make_fold_attn_o(seq, dk, bh, m, sb, kv_ratio)));
        };
        const auto combo_name = [](nn::expr::AttnMaskKind m, bool sb, std::uint32_t kv_ratio)
        {
            std::string n = m == nn::expr::AttnMaskKind::Plain    ? "plain"
                            : m == nn::expr::AttnMaskKind::Causal ? "causal"
                                                                  : "doc";
            if (sb) n += "+alibi";
            n += kv_ratio > 1 ? " gqa" : " mha";
            return n;
        };
        std::vector<std::string> keys;
        for (const nn::expr::AttnMaskKind m :
             {nn::expr::AttnMaskKind::Plain, nn::expr::AttnMaskKind::Causal,
              nn::expr::AttnMaskKind::CausalDoc})
            for (const bool sb : {false, true})
            {
                if (m == nn::expr::AttnMaskKind::Plain && sb)
                    continue; // 本仓不构造该组合（与自登记表同规则）
                for (const std::uint32_t kv_ratio : {1u, 2u})
                {
                    const std::string k = key_of(m, sb, kv_ratio, 64, 2, 2);
                    keys.push_back(k);
                    // 形状无关：另一组完全不同（含 > EXPR_FOLD_BLOCK 的 seq）
                    // 的形状 + 另一个 GQA 比必须折叠出同一 key。
                    check(k == key_of(m, sb, kv_ratio == 1 ? 1u : 4u, 512, 64, 8),
                          "attn fold " + combo_name(m, sb, kv_ratio) +
                              " 形状无关（key 跨形状不变）");
                }
            }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        check(keys.size() == 10, "attn fold 10 个组合（掩码 × 偏置 × {MHA,GQA}）互不相同 n=" +
                                     std::to_string(keys.size()));
    }

    // ── 9) 行并行门控覆盖（#T1）─────────────────────────────────────────
    // 门控在 compute_cpu_engine.hpp 的 eval_fold_impl：rows*K >= 65536 才按行
    // 分片并行（kFoldParallelMinWork），否则串行。此前所有 fold 用例都远低于
    // 该阈值（rows=5、K≤1000），并行分支从未被执行。本段：
    //   a) 真正越过门控（rows=64, K=4096 → 262144 ≥ 65536）并断言与"逐行
    //      串行"（rows=1 → 门控不成立）**逐位**一致；
    //   b) 门控边界（15×4096=61440 关 / 16×4096=65536 开）两侧前 15 行逐位相同。
    // 行间完全独立、行内逐位确定 ⇒ 并行结果必须与串行完全相等（不用容差）。
    {
        constexpr std::size_t K = 4096;
        const auto fill_rnd = [&](nn::Matrix& m)
        { for (auto& v : m.span()) v = dist(rng); };

        // 逐行求值：rows=1 使 rows*K=K < 65536，强制串行分支；行内容与整块相同。
        const auto eval_rows_serial = [&](const nn::ExprSpec& spec, const nn::Matrix& mx,
                                          std::size_t r_begin, std::size_t r_end,
                                          const char* tag)
        {
            std::vector<Scalar> got;
            got.reserve(r_end - r_begin);
            const auto src = mx.span();
            for (std::size_t r = r_begin; r < r_end; ++r)
            {
                nn::Matrix one(1, K);
                auto dst = one.span();
                for (std::size_t j = 0; j < K; ++j) dst[j] = src[r * K + j];
                nn::Tensor t = upload(engine, one);
                const std::array<nn::Tensor, 1> in{t};
                auto o = engine.eval_expr(spec, in, 1, 1);
                if (!o)
                {
                    check(false, std::string(tag) + " serial row " + std::to_string(r) +
                                     ": " + o.error().message);
                    return got;
                }
                const auto om = download(engine, *o);
                got.push_back(om.span()[0]);
            }
            return got;
        };
        const auto check_gate_parallel = [&](const nn::ExprSpec& spec, std::size_t rows,
                                             const char* tag)
        {
            nn::Matrix mx(rows, K);
            fill_rnd(mx);
            nn::Tensor x = upload(engine, mx);
            const std::array<nn::Tensor, 1> ins{x};
            auto o = engine.eval_expr(spec, ins, rows, 1);
            if (!o)
            {
                check(false, std::string(tag) + " parallel eval: " + o.error().message);
                return;
            }
            const auto om = download(engine, *o);
            const auto os = om.span();
            const auto ser = eval_rows_serial(spec, mx, 0, rows, tag);
            std::size_t first_bad = rows;
            if (ser.size() == rows)
                for (std::size_t r = 0; r < rows; ++r)
                    if (os[r] != ser[r])
                    {
                        first_bad = r;
                        break;
                    }
            check(ser.size() == rows && first_bad == rows,
                  std::string(tag) + " rows=" + std::to_string(rows) +
                      " K=" + std::to_string(K) + " rows*K=" + std::to_string(rows * K) +
                      "（门控内）并行 == 逐行串行（逐位）" +
                      (first_bad < rows ? "：row " + std::to_string(first_bad) + " 首次不等" : ""));
        };

        // 门控边界：同一份 16 行数据，15 行（61440 < 65536，关）与 16 行
        // （65536，开）各求值一次，比较两侧的前 15 行。
        nn::Matrix m16(16, K);
        fill_rnd(m16);
        const auto eval_prefix = [&](const nn::ExprSpec& spec, std::size_t nrows)
            -> std::optional<nn::Matrix>
        {
            nn::Matrix m(nrows, K);
            const auto src = m16.span();
            auto dst = m.span();
            for (std::size_t i = 0; i < nrows * K; ++i) dst[i] = src[i];
            nn::Tensor t = upload(engine, m);
            const std::array<nn::Tensor, 1> in{t};
            auto o = engine.eval_expr(spec, in, nrows, 1);
            if (!o) return std::nullopt;
            return download(engine, *o);
        };
        const auto check_gate_boundary = [&](const nn::ExprSpec& spec, const char* tag)
        {
            const auto off = eval_prefix(spec, 15); // 61440 < 65536
            const auto on = eval_prefix(spec, 16);  // 65536 == 阈值
            if (!off || !on)
            {
                check(false, std::string(tag) + " boundary eval failed");
                return;
            }
            const auto a = off->span();
            const auto b = on->span();
            bool same = true;
            std::size_t bad = 0;
            for (std::size_t r = 0; r < 15; ++r)
                if (a[r] != b[r])
                {
                    same = false;
                    bad = r;
                    break;
                }
            check(same, std::string(tag) +
                            " 门控边界 15×4096(关) vs 16×4096(开) 前 15 行逐位相同" +
                            (same ? "" : "：row " + std::to_string(bad) + " 首次不等"));
        };

        const nn::ExprSpec rowsum = nn::expr::make_fold_rowsum(static_cast<std::uint32_t>(K));
        const nn::ExprSpec sden =
            nn::expr::make_fold_softmax_denom(static_cast<std::uint32_t>(K));
        // 分片数 = min(线程池, rows/8)：池 <2 时门控虽开但 n_chunks=1（分支不可达）。
        // 打印实际分片数，避免"跑在单线程机上却误以为覆盖了并行"。
        const std::size_t workers = nn::global_thread_pool().size();
        std::cout << "    行并行门控：线程池 " << workers << " 线程，rows=64 → "
                  << std::min(workers, std::size_t{8}) << " 片（rows*K=262144 ≥ 65536）\n";
        check_gate_parallel(rowsum, 64, "rowsum");
        check_gate_parallel(sden, 64, "softmax_denom");
        check_gate_boundary(rowsum, "rowsum");
        check_gate_boundary(sden, "softmax_denom");
    }

    return fail;
}
