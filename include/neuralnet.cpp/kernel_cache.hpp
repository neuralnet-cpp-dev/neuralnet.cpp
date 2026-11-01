#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  kernel_cache.hpp — 运行期 JIT 算子缓存（kernels.nnk）
//
//  定位：`.nnk` 是**缓存**（torch.compile 式），不是部署代码模块。
//  记录：K = "v<生成器版本>.<expr_prec_sig_key(expr_spec_key, 所需签名)>" → V = SPIR-V 字节。
//  容器：KVRecord v2（KeyValueRecord），每条 kernel 一个字段 "kernel.<K>" → SPIR-V 字节
//        （key_index_ 使 has/get 按 key O(1) 命中；set + serialize_v2 支持运行期追加重建）。
//  版本失效：生成器版本进 K；引擎升级后 K 全变 → 旧条目永不匹配（孤立无害），
//        新 key 触发重编；加载时 generator_version 不符 → 判陈旧 → 丢弃重建。
//  设计见 docs/internal/33-kernel-cache-and-engine-decouple.md。
//
//  本头还含 ExprSpec 序列化器 —— 它**不是缓存值**，而是生成器 `.so`
//  （libnn-kernelgen.so，Phase 2）的 C ABI 跨边界载荷编码（doc §3.2）。
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <neuralnet.cpp/core_env.hpp>
#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/expr_spec.hpp>
#include <neuralnet.cpp/model_keyvalue_record.hpp>

namespace nn::fused
{

// 生成器版本：只在**代码生成语义变化**时 bump（非每次 release；避免无关升级
// 频繁作废缓存）。进 K 的版本组件与 .nnk 的 generator_version 字段。
inline constexpr const char* kKernelCacheGeneratorVersion = "1";
inline constexpr std::uint64_t kKernelCacheFormatVersion = 1;

// KVRecord 里每条 kernel 一个字段，前缀 "kernel." + K。
inline constexpr const char* kKernelFieldPrefix = "kernel.";

// K = "v<生成器版本>.<表达式签名 key>"
// 表达式签名 key：V0（sig=0）= 结构 key；分派哨兵 = key + "#x"（EXPR_PREC_DISPATCH_SUFFIX，
// 与运行期 dispatch 查找一致）；静态签名 = expr_prec_sig_key（key + "#xxxx"）。
[[nodiscard]] inline std::string kernel_cache_key(const std::string& spec_key,
                                                  std::uint32_t prec_sig)
{
    std::string body = spec_key;
    if (nn::expr_prec_sig_is_dispatch(prec_sig))
        body += nn::EXPR_PREC_DISPATCH_SUFFIX;
    else
        body = nn::expr_prec_sig_key(spec_key, prec_sig);
    return "v" + std::string(kKernelCacheGeneratorVersion) + "." + body;
}

// ── 缓存位置（非缓存版，供测试多次调）────────────────────────────────────
// 优先级：NN_KERNEL_CACHE_PATH（显式文件）> NN_KERNEL_CACHE_DIR（目录）>
// 系统统一缓存目录 + "neuralnet-cpp/kernels.nnk"。路径分隔统一用 '/'（Windows
// CRT 接受，跨平台一致）。
[[nodiscard]] inline std::string kernel_cache_path_from_env()
{
    // 1) 显式文件路径
    if (const std::string p = nn::detail::env_string_raw("NN_KERNEL_CACHE_PATH"); !p.empty())
        return p;
    // 2) 显式缓存目录
    if (const std::string d = nn::detail::env_string_raw("NN_KERNEL_CACHE_DIR"); !d.empty())
        return d + "/kernels.nnk";
    // 3) 系统统一缓存目录 + 库子目录
    std::string base;
#if defined(_WIN32)
    base = nn::detail::env_string_raw("LOCALAPPDATA");
    if (base.empty()) base = nn::detail::env_string_raw("USERPROFILE") + "/AppData/Local";
#elif defined(__APPLE__)
    base = nn::detail::env_string_raw("HOME") + "/Library/Caches";
#else
    base = nn::detail::env_string_raw("XDG_CACHE_HOME");
    if (base.empty()) base = nn::detail::env_string_raw("HOME") + "/.cache";
#endif
    if (base.empty()) base = ".";  // 都取不到 → 回退当前目录
    return base + "/neuralnet-cpp/kernels.nnk";
}
// 运行期取缓存路径：**读一次 + 缓存**（环境变量行为开关，见 core_env 注释）。
[[nodiscard]] inline const std::string& kernel_cache_path()
{
    static const std::string path = kernel_cache_path_from_env();
    return path;
}

// ── ExprSpec 序列化（生成器 .so 的 C ABI 边界载荷；见 doc §3.2）──────────
// 与缓存值无关；Phase 1 保留并测试，Phase 2 作为 libnn-kernelgen.so 的接口契约。
namespace detail
{

struct ByteWriter
{
    std::vector<std::uint8_t> buf;
    void u8(std::uint8_t v) { buf.push_back(v); }
    void u32(std::uint32_t v)
    {
        buf.push_back(static_cast<std::uint8_t>(v & 0xffu));
        buf.push_back(static_cast<std::uint8_t>((v >> 8) & 0xffu));
        buf.push_back(static_cast<std::uint8_t>((v >> 16) & 0xffu));
        buf.push_back(static_cast<std::uint8_t>((v >> 24) & 0xffu));
    }
    void f32(float v)
    {
        std::uint32_t x;
        std::memcpy(&x, &v, sizeof(x));
        u32(x);
    }
    void raw_bytes(const void* p, std::size_t n)
    {
        const auto* b = static_cast<const std::uint8_t*>(p);
        buf.insert(buf.end(), b, b + n);
    }
    void len_bytes(const void* p, std::size_t n)
    {
        u32(static_cast<std::uint32_t>(n));
        raw_bytes(p, n);
    }
};

struct ByteReader
{
    const std::uint8_t* p = nullptr;
    std::size_t n = 0;
    std::size_t pos = 0;
    bool fail = false;

    bool need(std::size_t m)
    {
        if (fail || pos + m > n) { fail = true; return false; }
        return true;
    }
    bool u8(std::uint8_t& v)
    {
        if (!need(1)) return false;
        v = p[pos++];
        return true;
    }
    bool u32(std::uint32_t& v)
    {
        if (!need(4)) return false;
        v = static_cast<std::uint32_t>(p[pos]) |
            (static_cast<std::uint32_t>(p[pos + 1]) << 8) |
            (static_cast<std::uint32_t>(p[pos + 2]) << 16) |
            (static_cast<std::uint32_t>(p[pos + 3]) << 24);
        pos += 4;
        return true;
    }
    bool f32(float& v)
    {
        std::uint32_t x;
        if (!u32(x)) return false;
        std::memcpy(&v, &x, sizeof(v));
        return true;
    }
    bool len_bytes(std::vector<std::uint8_t>& v)
    {
        std::uint32_t len;
        if (!u32(len)) return false;
        if (!need(len)) return false;
        v.assign(p + pos, p + pos + len);
        pos += len;
        return true;
    }
};

inline void write_f32_vec(ByteWriter& w, const std::vector<Scalar>& v)
{
    w.u32(static_cast<std::uint32_t>(v.size()));
    for (const float x : v) w.f32(x);
}
inline bool read_f32_vec(ByteReader& r, std::vector<Scalar>& v)
{
    std::uint32_t len;
    if (!r.u32(len)) return false;
    v.resize(len);
    for (auto& x : v)
        if (!r.f32(x)) return false;
    return true;
}

inline void write_operand(ByteWriter& w, const ExprOperand& o)
{
    w.u8(o.kind);
    w.u8(o.idx);
}
inline bool read_operand(ByteReader& r, ExprOperand& o)
{
    return r.u8(o.kind) && r.u8(o.idx);
}
inline void write_view(ByteWriter& w, const ExprView& v)
{
    w.u8(v.kind);
    w.u8(v.negate_first_half);
    w.u32(v.param);
    w.u32(v.param2);
}
inline bool read_view(ByteReader& r, ExprView& v)
{
    return r.u8(v.kind) && r.u8(v.negate_first_half) && r.u32(v.param) && r.u32(v.param2);
}
inline void write_instr(ByteWriter& w, const ExprInstr& in)
{
    w.u8(in.op);
    w.u8(in.dst);
    write_operand(w, in.a);
    write_operand(w, in.b);
    write_operand(w, in.c);
}
inline bool read_instr(ByteReader& r, ExprInstr& in)
{
    return r.u8(in.op) && r.u8(in.dst) && read_operand(r, in.a) && read_operand(r, in.b) &&
           read_operand(r, in.c);
}
inline void write_instr_seq(ByteWriter& w, const std::vector<ExprInstr>& seq)
{
    w.u32(static_cast<std::uint32_t>(seq.size()));
    for (const auto& in : seq) write_instr(w, in);
}
inline bool read_instr_seq(ByteReader& r, std::vector<ExprInstr>& seq)
{
    std::uint32_t len;
    if (!r.u32(len)) return false;
    seq.resize(len);
    for (auto& in : seq)
        if (!read_instr(r, in)) return false;
    return true;
}
inline void write_matmul(ByteWriter& w, const MatmulSpec& m)
{
    w.u8(m.a_input);
    w.u8(m.b_input);
    w.u8(m.transA);
    w.u8(m.transB);
    w.u32(m.k);
    w.u32(m.batch);
}
inline bool read_matmul(ByteReader& r, MatmulSpec& m)
{
    return r.u8(m.a_input) && r.u8(m.b_input) && r.u8(m.transA) && r.u8(m.transB) &&
           r.u32(m.k) && r.u32(m.batch);
}
inline void write_vecacc(ByteWriter& w, const VecAccSpec& v)
{
    w.u8(v.vec_state);
    w.u8(v.weight_reg);
    w.u8(v.b_input);
    w.u8(v.scale_reg);
    w.u8(v.has_scale);
}
inline bool read_vecacc(ByteReader& r, VecAccSpec& v)
{
    return r.u8(v.vec_state) && r.u8(v.weight_reg) && r.u8(v.b_input) && r.u8(v.scale_reg) &&
           r.u8(v.has_scale);
}

inline void write_fold(ByteWriter& w, const FoldSpec& f)
{
    w.u8(f.num_state);
    w.u32(f.k);
    write_f32_vec(w, f.inits);
    write_instr_seq(w, f.body);
    write_instr_seq(w, f.finalize);
    w.u32(f.vec_state_len);
    w.u8(f.matmul ? 1 : 0);
    if (f.matmul) write_matmul(w, *f.matmul);
    w.u8(f.vecacc ? 1 : 0);
    if (f.vecacc) write_vecacc(w, *f.vecacc);
    w.u8(f.tri_skip ? 1 : 0);
}
inline bool read_fold(ByteReader& r, FoldSpec& f)
{
    if (!r.u8(f.num_state) || !r.u32(f.k) || !read_f32_vec(r, f.inits) ||
        !read_instr_seq(r, f.body) || !read_instr_seq(r, f.finalize))
        return false;
    if (!r.u32(f.vec_state_len)) return false;
    std::uint8_t has_mm, has_va, tri;
    if (!r.u8(has_mm)) return false;
    if (has_mm)
    {
        MatmulSpec m;
        if (!read_matmul(r, m)) return false;
        f.matmul = m;
    }
    if (!r.u8(has_va)) return false;
    if (has_va)
    {
        VecAccSpec va;
        if (!read_vecacc(r, va)) return false;
        f.vecacc = va;
    }
    if (!r.u8(tri)) return false;
    f.tri_skip = tri != 0;
    return true;
}

inline void write_expr_spec(ByteWriter& w, const ExprSpec& s)
{
    write_instr_seq(w, s.instrs);
    w.u32(static_cast<std::uint32_t>(s.views.size()));
    for (const auto& v : s.views) write_view(w, v);
    write_f32_vec(w, s.consts);
    write_f32_vec(w, s.rparams);
    w.u32(s.num_regs);
    w.u8(s.matmul ? 1 : 0);
    if (s.matmul) write_matmul(w, *s.matmul);
    w.u8(s.fold ? 1 : 0);
    if (s.fold) write_fold(w, *s.fold);
    w.u32(static_cast<std::uint32_t>(s.extras.size()));
    for (const auto e : s.extras) w.u8(e);
}
inline bool read_expr_spec(ByteReader& r, ExprSpec& s)
{
    if (!read_instr_seq(r, s.instrs)) return false;
    std::uint32_t nviews;
    if (!r.u32(nviews)) return false;
    s.views.resize(nviews);
    for (auto& v : s.views)
        if (!read_view(r, v)) return false;
    if (!read_f32_vec(r, s.consts) || !read_f32_vec(r, s.rparams)) return false;
    if (!r.u32(s.num_regs)) return false;
    std::uint8_t has_mm, has_fold;
    if (!r.u8(has_mm)) return false;
    if (has_mm)
    {
        MatmulSpec m;
        if (!read_matmul(r, m)) return false;
        s.matmul = m;
    }
    if (!r.u8(has_fold)) return false;
    if (has_fold)
    {
        FoldSpec f;
        if (!read_fold(r, f)) return false;
        s.fold = f;
    }
    std::uint32_t nextras;
    if (!r.u32(nextras)) return false;
    s.extras.resize(nextras);
    for (auto& e : s.extras)
        if (!r.u8(e)) return false;
    return true;
}

} // namespace detail

// 序列化 ExprSpec 为字节（生成器 .so 边界载荷；round-trip 逐位一致，见 test）
[[nodiscard]] inline std::vector<std::uint8_t> encode_expr_spec(const ExprSpec& s)
{
    detail::ByteWriter w;
    detail::write_expr_spec(w, s);
    return std::move(w.buf);
}
[[nodiscard]] inline Result<ExprSpec> decode_expr_spec(const std::uint8_t* data, std::size_t n)
{
    detail::ByteReader r{data, n, 0, false};
    ExprSpec s;
    if (!detail::read_expr_spec(r, s))
        NN_FAIL("kernel_cache: ExprSpec 解码失败（截断/损坏）");
    return s;
}

// ── KVRecord v2 缓存容器 ───────────────────────────────────────────────
// 编码：单个 KeyValueRecord（v2），字段 = nnk_format_version / generator_version /
//   "kernel.<K>" = SPIR-V 字节（string 字段，length-prefixed，任意字节安全）。
[[nodiscard]] inline std::string encode_kernel_cache(
    const std::unordered_map<std::string, std::vector<std::uint32_t>>& kernels,
    const std::string& generator_version)
{
    KeyValueRecord rec;
    rec.set("nnk_format_version", kKernelCacheFormatVersion);
    rec.set("generator_version", generator_version);
    for (const auto& [K, spirv] : kernels)
    {
        const std::string bytes(reinterpret_cast<const char*>(spirv.data()),
                                spirv.size() * sizeof(std::uint32_t));
        rec.set(std::string(kKernelFieldPrefix) + K, bytes);
    }
    return rec.serialize_v2();
}

// 运行期 JIT 算子缓存：懒加载 + 内存热缓存 + 追加持久化。
struct KernelCache
{
    std::string generator_version;
    KeyValueRecord rec;                       // 全量 kernel（持久化映像，懒查询）
    std::unordered_map<std::string, std::vector<std::uint32_t>> hot;  // 内存热缓存

    // 加载：generator_version 与期望一致 → 复用；不符 → 判陈旧 → 空缓存（重建）。
    [[nodiscard]] static Result<KernelCache> from_bytes(std::string_view bytes,
                                                        const std::string& expected_genver)
    {
        auto rec_r = KeyValueRecord::parse(bytes);
        if (!rec_r) return std::unexpected(rec_r.error());
        KernelCache c;
        std::string genver;
        c.generator_version = (rec_r->get("generator_version", genver))
                                  ? genver
                                  : std::string(kKernelCacheGeneratorVersion);
        if (c.generator_version == expected_genver)
        {
            c.rec = std::move(*rec_r);
        }
        else
        {
            // 陈旧：丢弃旧缓存（字段不读、不命中），换新生成器版本重建。
            c.generator_version = expected_genver;
            c.rec.set("nnk_format_version", kKernelCacheFormatVersion);
            c.rec.set("generator_version", expected_genver);
        }
        return c;
    }

    // 命中：内存热缓存；内存 miss → 从 rec 懒加载（O(1)，key_index_）。返回 nullptr = miss。
    [[nodiscard]] const std::vector<std::uint32_t>* hit(const std::string& K)
    {
        const auto it = hot.find(K);
        if (it != hot.end()) return &it->second;
        std::string bytes;
        if (rec.get(std::string(kKernelFieldPrefix) + K, bytes))
        {
            const auto* data = reinterpret_cast<const std::uint32_t*>(bytes.data());
            std::vector<std::uint32_t> spv(data, data + bytes.size() / sizeof(std::uint32_t));
            const auto [ins, _] = hot.emplace(K, std::move(spv));
            return &ins->second;
        }
        return nullptr;
    }

    // JIT miss 后追加：内存 + rec（调用方随后 serialize()+ 写盘持久化）。
    [[nodiscard]] Result<void> insert(const std::string& K,
                                      const std::vector<std::uint32_t>& spirv)
    {
        hot[K] = spirv;
        const std::string bytes(reinterpret_cast<const char*>(spirv.data()),
                                spirv.size() * sizeof(std::uint32_t));
        rec.set(std::string(kKernelFieldPrefix) + K, bytes);
        return {};
    }

    // 当前 rec 序列化（供写盘持久化）
    [[nodiscard]] std::string serialize() const { return rec.serialize_v2(); }

    // 合并另一份 rec 里本 rec 没有的 kernel 字段（进程锁写盘前，把并发进程中
    // 其它进程已 JIT 并落盘的 kernel 合进来，避免 lost-update）。hot 不更新
    // （hit() 懒加载自 rec，合进来的条目后续按需进 hot）。
    [[nodiscard]] Result<void> merge_kernels_from(const KeyValueRecord& other)
    {
        for (const std::string& key : other.string_keys_with_prefix(kKernelFieldPrefix))
        {
            std::string v;
            if (!other.get(key, v)) continue;
            std::string mine;
            if (!rec.get(key, mine)) rec.set(key, v);
        }
        return {};
    }
};

} // namespace nn::fused
