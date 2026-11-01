// ── kernel_cache_test — 可移植算子缓存（kernels.nnk）序列化 round-trip 逐位锁 ──
// 背景（docs/internal/33）：AOT 融合注册表解耦为可移植二进制缓存 .nnk。
//   D5 = 必须序列化**完整 ExprSpec**（运行期 IR-C 图融合读 k.spec.fold/extras/views）。
//   序列化器漏字段 → 图融合读到未初始化值 = 静默错值。
// 本测试锁：
//   ① 各类 ExprSpec（逐元素 / matmul / fold v1 / fold v2 双域 / extras / consts+rparams /
//      RowAccess 等运行时参数视图）encode → decode 后与内存对象**逐位一致**（expr_spec_equal）
//   ② 整文件 .nnk：校验和通过、条目数/顺序/每条目的 spirv 与 meta 逐位一致
//   ③ 负例：magic 不符 / 截断 / 校验和不匹配 → decode 返回错误（不崩溃、不静默放行）
//
// 纯 CPU、无 GPU 依赖。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <neuralnet.cpp/kernel_cache.hpp>

namespace
{
int g_fail = 0;

void check(bool ok, const char* what)
{
    if (ok)
    {
        std::printf("[PASS] %s\n", what);
        return;
    }
    ++g_fail;
    std::printf("[FAIL] %s\n", what);
}

// 构建一组覆盖各形态的 ExprSpec（round-trip 只关心 encode/decode 逐位一致，
// 语义正确性由融合路径测试负责；此处用手工结构即可）。
nn::ExprSpec spec_elem()
{
    nn::ExprSpec s;
    s.instrs = {{static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                 {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 0},
                 {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 1}, {0, 0}}};
    s.views = {{0, 0, 0, 0}, {0, 0, 0, 0}};  // 两条 Linear
    s.num_regs = 1;
    return s;
}

nn::ExprSpec spec_const_rparam()
{
    nn::ExprSpec s;
    s.instrs = {
        {static_cast<std::uint8_t>(nn::ExprOp::Mul), 0,
         {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 0},
         {static_cast<std::uint8_t>(nn::ExprOperandKind::Const), 0}, {0, 0}},
        {static_cast<std::uint8_t>(nn::ExprOp::Add), 1,
         {static_cast<std::uint8_t>(nn::ExprOperandKind::Reg), 0},
         {static_cast<std::uint8_t>(nn::ExprOperandKind::RParam), 0}, {0, 0}},
    };
    s.views = {{0, 0, 0, 0}};
    s.consts = {2.0f};
    s.rparams = {0.5f};
    s.num_regs = 2;
    return s;
}

nn::ExprSpec spec_matmul()
{
    nn::ExprSpec s;
    nn::MatmulSpec m;
    m.a_input = 0;
    m.b_input = 1;
    m.transA = 0;
    m.transB = 1;
    m.k = 64;
    m.batch = 2;
    s.matmul = m;  // instrs 空 → 输出 = matmul 结果
    s.num_regs = 0;
    return s;
}

nn::ExprSpec spec_fold_v1()
{
    nn::ExprSpec s;
    nn::FoldSpec f;
    f.num_state = 1;
    f.k = 32;
    f.inits = {0.0f};
    f.body = {{static_cast<std::uint8_t>(nn::ExprOp::RowSum), 1,
               {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 0}, {0, 0}, {0, 0}}};
    f.finalize = {{static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                   {static_cast<std::uint8_t>(nn::ExprOperandKind::Reg), 0},
                   {static_cast<std::uint8_t>(nn::ExprOperandKind::Reg), 1}, {0, 0}}};
    f.vec_state_len = 0;
    f.tri_skip = true;
    s.fold = f;
    s.num_regs = 2;
    return s;
}

nn::ExprSpec spec_fold_v2()
{
    nn::ExprSpec s;
    nn::FoldSpec f;
    f.num_state = 1;
    f.k = 16;
    f.inits = {1.0f};
    f.body = {{static_cast<std::uint8_t>(nn::ExprOp::ColSum), 1,
               {static_cast<std::uint8_t>(nn::ExprOperandKind::Matmul), 0}, {0, 0}, {0, 0}}};
    f.finalize = {{static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                   {static_cast<std::uint8_t>(nn::ExprOperandKind::Reg), 0},
                   {static_cast<std::uint8_t>(nn::ExprOperandKind::VecState), 0}, {0, 0}}};
    f.vec_state_len = 8;
    nn::MatmulSpec m;
    m.a_input = 0;
    m.b_input = 1;
    m.transA = 0;
    m.transB = 0;
    m.k = 16;
    m.batch = 1;
    f.matmul = m;
    nn::VecAccSpec va;
    va.vec_state = 0;
    va.weight_reg = 1;
    va.b_input = 2;
    va.scale_reg = 3;
    va.has_scale = 1;
    f.vecacc = va;
    s.fold = f;
    s.num_regs = 2;
    return s;
}

nn::ExprSpec spec_extras()
{
    nn::ExprSpec s;
    s.instrs = {
        {static_cast<std::uint8_t>(nn::ExprOp::Mul), 0,
         {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 0},
         {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 1}, {0, 0}},
        {static_cast<std::uint8_t>(nn::ExprOp::Add), 1,
         {static_cast<std::uint8_t>(nn::ExprOperandKind::Reg), 0},
         {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 2}, {0, 0}},
    };
    s.views = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    s.extras = {0};  // 主输出 = dst1；额外物化 dst0
    s.num_regs = 2;
    return s;
}

nn::ExprSpec spec_runtime_views()
{
    nn::ExprSpec s;
    s.instrs = {{static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                 {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 0},
                 {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 1}, {0, 0}}};
    // RotateHalf（param=block_rows）+ RowAccess（mod + offset 两个 vp 槽）
    s.views = {{static_cast<std::uint8_t>(nn::ExprViewKind::RotateHalf), 1, 64, 0},
               {static_cast<std::uint8_t>(nn::ExprViewKind::RowAccess), 0, 128, 32}};
    s.num_regs = 1;
    return s;
}

void roundtrip_one(const nn::ExprSpec& s, const char* name)
{
    nn::fused::FusedKernel k;
    k.key = "test#" + std::string(name);
    k.spec = s;
    k.spirv = {0x07230203u, 0x00010000u, 0x0000000Bu};
    k.reduce_axis = -1;
    k.has_matmul = s.matmul ? 1 : 0;
    k.view_param_count = 3;
    k.rparam_count = static_cast<std::uint32_t>(s.rparams.size());
    k.vec_width = 4;
    k.prec_sig = 0x00020000u;

    nn::fused::detail::ByteWriter w;
    nn::fused::encode_kernel(w, k);
    nn::fused::detail::ByteReader r{w.buf.data(), w.buf.size(), 0, false};
    nn::fused::FusedKernel k2;
    const bool ok_decode = nn::fused::decode_kernel(r, k2);

    char label[64];
    std::snprintf(label, sizeof(label), "%s：decode 成功", name);
    check(ok_decode, label);
    check(k2.key == k.key, (std::string(name) + "：key 一致").c_str());
    check(k2.spirv == k.spirv, (std::string(name) + "：spirv 一致").c_str());
    check(nn::expr_spec_equal(k2.spec, k.spec),
          (std::string(name) + "：ExprSpec 逐位一致（expr_spec_equal）").c_str());
    check(k2.reduce_axis == k.reduce_axis && k2.has_matmul == k.has_matmul &&
              k2.view_param_count == k.view_param_count &&
              k2.rparam_count == k.rparam_count && k2.vec_width == k.vec_width &&
              k2.prec_sig == k.prec_sig,
          (std::string(name) + "：meta 一致").c_str());
}

void test_roundtrip_specs()
{
    roundtrip_one(spec_elem(), "elem");
    roundtrip_one(spec_const_rparam(), "const_rparam");
    roundtrip_one(spec_matmul(), "matmul");
    roundtrip_one(spec_fold_v1(), "fold_v1");
    roundtrip_one(spec_fold_v2(), "fold_v2");
    roundtrip_one(spec_extras(), "extras");
    roundtrip_one(spec_runtime_views(), "runtime_views");
}

void test_full_cache()
{
    std::vector<nn::fused::FusedKernel> kernels;
    {
        nn::fused::FusedKernel k;
        k.key = "k1";
        k.spec = spec_elem();
        k.spirv = {1, 2, 3};
        kernels.push_back(k);
    }
    {
        nn::fused::FusedKernel k;
        k.key = "k2#a";
        k.spec = spec_fold_v2();
        k.spirv = {7, 8};
        k.prec_sig = 0x00010001u;
        kernels.push_back(k);
    }

    const auto bytes = nn::fused::encode_kernel_cache(kernels);
    check(!bytes.empty(), "整文件 .nnk 编码非空");

    auto dec_r = nn::fused::decode_kernel_cache(bytes.data(), bytes.size());
    check(dec_r.has_value(), "整文件 .nnk 解码成功（校验和通过）");
    if (dec_r)
    {
        check(dec_r->size() == 2, "条目数一致（2）");
        if (dec_r->size() == 2)
        {
            check((*dec_r)[0].key == "k1" && (*dec_r)[0].spirv == std::vector<std::uint32_t>{1, 2, 3},
                  "条目 0 key/spirv 一致");
            check(nn::expr_spec_equal((*dec_r)[0].spec, spec_elem()), "条目 0 ExprSpec 一致");
            check((*dec_r)[1].key == "k2#a" && (*dec_r)[1].prec_sig == 0x00010001u,
                  "条目 1 key/prec_sig 一致");
        }
    }

    // KernelCache 加载器（index 命中）
    auto cache_r = nn::fused::KernelCache::from_bytes(bytes.data(), bytes.size());
    check(cache_r.has_value(), "KernelCache::from_bytes 成功");
    if (cache_r)
    {
        check(cache_r->hit("k1") != nullptr && cache_r->hit("k1")->spirv == std::vector<std::uint32_t>{1, 2, 3},
              "hit(\"k1\") 命中");
        check(cache_r->hit("nope") == nullptr, "hit(\"nope\") 未命中");
        check(cache_r->contains("k2#a"), "contains(\"k2#a\")");
    }
}

void test_negative()
{
    auto good = nn::fused::encode_kernel_cache(
        [] {
            std::vector<nn::fused::FusedKernel> v;
            nn::fused::FusedKernel k;
            k.key = "x";
            k.spec = spec_elem();
            k.spirv = {9};
            v.push_back(k);
            return v;
        }());

    // magic 不符
    {
        auto bad = good;
        bad[0] = 'X';  // 破坏 "NNKFUSED" 首字节
        auto r = nn::fused::decode_kernel_cache(bad.data(), bad.size());
        check(!r.has_value(), "负例：magic 破坏 → 报错");
    }
    // 截断（去掉 trailer）
    {
        auto bad = std::vector<std::uint8_t>(good.begin(), good.end() - 1);
        auto r = nn::fused::decode_kernel_cache(bad.data(), bad.size());
        check(!r.has_value(), "负例：截断 → 报错");
    }
    // 校验和不匹配（破坏条目区中间一个字节）
    {
        auto bad = good;
        const std::size_t mid = good.size() / 2;
        bad[mid] ^= 0xFFu;
        auto r = nn::fused::decode_kernel_cache(bad.data(), bad.size());
        check(!r.has_value(), "负例：内容破坏 → 校验和报错");
    }
}
}  // namespace

int main()
{
    std::printf("kernel_cache_test: ExprSpec 序列化 round-trip + 整文件 .nnk 逐位锁\n\n");
    test_roundtrip_specs();
    test_full_cache();
    test_negative();
    std::printf("\nkernel_cache_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
