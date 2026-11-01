// ── kernel_cache_test — 可移植算子缓存（kernels.nnk，KVRecord v2）round-trip 锁 ──
// 背景（docs/internal/33）：`.nnk` 是运行期 JIT 算子缓存，`K=v<生成器版本>.<表达式>` → `V=SPIR-V`，
// 用 KVRecord v2 承载。本测试锁：
//   ① ExprSpec 序列化器 round-trip 逐位一致（生成器 .so 的 C ABI 边界载荷，doc §3.2）
//   ② KVRecord v2 缓存：encode → from_bytes → hit（含内存懒加载）、insert 追加、serialize 回环
//   ③ 版本失效：generator_version 不符 → 判陈旧 → 空缓存（旧 kernel 不命中）
//   ④ kernel_cache_key 编码 = "v<版本>.<expr_prec_sig_key(expr_spec_key, sig)>"
//
// 纯 CPU、无 GPU 依赖。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
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

// ── 一组覆盖各形态的 ExprSpec（round-trip 只关心 encode/decode 逐位一致）──
nn::ExprSpec spec_elem()
{
    nn::ExprSpec s;
    s.instrs = {{static_cast<std::uint8_t>(nn::ExprOp::Add), 0,
                 {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 0},
                 {static_cast<std::uint8_t>(nn::ExprOperandKind::Input), 1}, {0, 0}}};
    s.views = {{0, 0, 0, 0}, {0, 0, 0, 0}};
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
    m.transB = 1;
    m.k = 64;
    m.batch = 2;
    s.matmul = m;
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
    m.k = 16;
    f.matmul = m;
    nn::VecAccSpec va;
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
    s.extras = {0};
    s.num_regs = 2;
    return s;
}

void test_expr_spec_roundtrip()
{
    const nn::ExprSpec specs[] = {spec_elem(),       spec_const_rparam(), spec_matmul(),
                                  spec_fold_v1(),   spec_fold_v2(),      spec_extras()};
    const char* names[] = {"elem", "const_rparam", "matmul", "fold_v1", "fold_v2", "extras"};
    for (int i = 0; i < 6; ++i)
    {
        const auto bytes = nn::fused::encode_expr_spec(specs[i]);
        auto r = nn::fused::decode_expr_spec(bytes.data(), bytes.size());
        char label[64];
        std::snprintf(label, sizeof(label), "%s：ExprSpec round-trip 逐位一致", names[i]);
        check(r.has_value() && nn::expr_spec_equal(*r, specs[i]), label);
    }
}

void test_cache_roundtrip()
{
    std::unordered_map<std::string, std::vector<std::uint32_t>> kernels;
    kernels["v1.aaa"] = {0x07230203u, 0x00010000u, 0x0000000Bu};
    kernels["v1.bbb#x"] = {7u, 8u, 9u};
    const auto bytes = nn::fused::encode_kernel_cache(kernels, "1");

    auto c_r = nn::fused::KernelCache::from_bytes(bytes, "1");
    check(c_r.has_value(), "KVRecord v2 缓存 from_bytes 成功（版本一致）");
    if (!c_r) return;
    auto& c = *c_r;
    check(c.generator_version == "1", "generator_version 读回为 1");
    check(c.hot.empty(), "加载后内存热缓存为空（懒加载）");

    // 命中：内存懒加载自 rec
    const auto* spv = c.hit("v1.aaa");
    check(spv != nullptr && *spv == std::vector<std::uint32_t>{0x07230203u, 0x00010000u, 0x0000000Bu},
          "hit(\"v1.aaa\") 从 rec 懒加载命中");
    check(c.hit("v1.aaa") == spv, "二次 hit 走内存热缓存（同一指针）");
    check(c.hit("nope") == nullptr, "hit(\"nope\") 未命中");
    check(c.hit("v1.bbb#x") != nullptr, "hit(\"v1.bbb#x\") 命中");

    // insert 追加 + serialize 回环
    auto ins = c.insert("v1.ccc#a", std::vector<std::uint32_t>{11u, 12u});
    check(ins.has_value() && c.hit("v1.ccc#a") != nullptr, "insert 追加 + 命中");
    const auto bytes2 = c.serialize();
    auto c2_r = nn::fused::KernelCache::from_bytes(bytes2, "1");
    check(c2_r.has_value() && c2_r->hit("v1.ccc#a") != nullptr,
          "insert 后 serialize 回环：追加条目可再加载");
}

void test_version_invalidation()
{
    // 用版本 "1" 编码的缓存，用版本 "2" 加载 → 判陈旧 → 空缓存
    std::unordered_map<std::string, std::vector<std::uint32_t>> kernels;
    kernels["v1.aaa"] = {1u, 2u, 3u};
    const auto bytes = nn::fused::encode_kernel_cache(kernels, "1");

    auto c_r = nn::fused::KernelCache::from_bytes(bytes, "2");
    check(c_r.has_value(), "版本不符：from_bytes 仍成功（但判陈旧）");
    if (!c_r) return;
    auto& c = *c_r;
    check(c.generator_version == "2", "陈旧后 generator_version 换为新版本 2");
    check(c.hit("v1.aaa") == nullptr, "陈旧缓存：旧 kernel 不命中（防读旧引擎 SPIR-V）");
    // 新版本下 JIT 重编 → insert 进新缓存
    check(c.insert("v2.aaa", std::vector<std::uint32_t>{9u, 9u}).has_value() &&
              c.hit("v2.aaa") != nullptr,
          "新版本下可写入新 kernel");
}

void test_key_encoding()
{
    // kernel_cache_key = "v<生成器版本>.<expr_prec_sig_key(expr_spec_key, sig)>"
    const std::string spec_key = nn::expr_spec_key(spec_elem());
    const auto k_disp = nn::fused::kernel_cache_key(spec_key, nn::EXPR_PREC_SIG_DISPATCH);
    const auto k_v0 = nn::fused::kernel_cache_key(spec_key, 0u);
    check(k_disp == "v" + std::string(nn::fused::kKernelCacheGeneratorVersion) + "." + spec_key +
                        nn::EXPR_PREC_DISPATCH_SUFFIX,
          "K 含生成器版本 + 表达式 + 分派后缀");
    check(k_v0 == "v" + std::string(nn::fused::kKernelCacheGeneratorVersion) + "." + spec_key,
          "K（V0）= 版本 + 表达式");
    check(k_disp != k_v0, "不同签名 → 不同 K");
}

// 环境变量路径解析（非缓存版，可多次调）。setenv 用标准接口。
void test_cache_path()
{
    // NN_KERNEL_CACHE_PATH 显式文件优先
    ::setenv("NN_KERNEL_CACHE_PATH", "/tmp/kc/custom.nnk", 1);
    ::setenv("NN_KERNEL_CACHE_DIR", "/tmp/kc/dir", 1);
    check(nn::fused::kernel_cache_path_from_env() == "/tmp/kc/custom.nnk",
          "NN_KERNEL_CACHE_PATH 优先");

    // 只设 NN_KERNEL_CACHE_DIR → 目录 + kernels.nnk
    ::unsetenv("NN_KERNEL_CACHE_PATH");
    check(nn::fused::kernel_cache_path_from_env() == "/tmp/kc/dir/kernels.nnk",
          "NN_KERNEL_CACHE_DIR → 目录/kernels.nnk");

    // 都不设 → 回退系统缓存目录（非空、以 kernels.nnk 结尾）
    ::unsetenv("NN_KERNEL_CACHE_DIR");
    const std::string def = nn::fused::kernel_cache_path_from_env();
    check(!def.empty() && def.size() >= std::string("kernels.nnk").size() &&
              def.substr(def.size() - std::string("kernels.nnk").size()) == "kernels.nnk",
          ("默认 → 系统缓存目录 kernels.nnk（" + def + "）").c_str());
}
}  // namespace

int main()
{
    std::printf("kernel_cache_test: ExprSpec round-trip + KVRecord v2 算子缓存（K→SPIR-V）\n\n");
    test_expr_spec_roundtrip();
    test_cache_roundtrip();
    test_version_invalidation();
    test_key_encoding();
    test_cache_path();
    std::printf("\nkernel_cache_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
