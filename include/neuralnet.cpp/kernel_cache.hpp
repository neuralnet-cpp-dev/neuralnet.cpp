#pragma once

// ═══════════════════════════════════════════════════════════════════════════
//  kernel_cache.hpp — 可移植算子缓存（kernels.nnk）格式与加载器
//
//  把 AOT 融合注册表从"构建期内嵌头（fused_registry.hpp）"解耦为可移植
//  二进制缓存文件，C++ 构建期生成、引擎运行期加载，Python 复用同一数据契约。
//  设计见 docs/internal/33-kernel-cache-and-engine-decouple.md。
//
//  本头只含纯数据/序列化逻辑（无引擎依赖，仅 <cstdint>/<vector>/<string>
//  + core_errors / core_sha256 / expr_spec），写入侧（tools/fused_generate）
//  与加载侧（引擎运行期）共用同一套 encode/decode。
//
//  .nnk 布局（little-endian）：
//    magic    8B   "NNKFUSED"
//    version  u32  1
//    flags    u32  位 0 = 仅 GPU 融合算子（本版恒 1）
//    count    u32  条目数
//    [条目 × count]
//      key       u32 len + bytes        （expr_spec_key，含变体后缀）
//      spec      u32 len + bytes        完整 ExprSpec 序列化
//      spirv     u32 words + u32[]      SPIR-V 二进制
//      meta      6 × u32                reduce_axis / has_matmul /
//                                       view_param_count / rparam_count /
//                                       vec_width / prec_sig
//    sha256  32B   整文件校验和（不含 sha256 与 trailer）
//    trailer  5B   "NNEND"
// ═══════════════════════════════════════════════════════════════════════════

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <neuralnet.cpp/core_errors.hpp>
#include <neuralnet.cpp/core_sha256.hpp>
#include <neuralnet.cpp/expr_spec.hpp>

namespace nn::fused
{

inline constexpr const char* kKernelCacheMagic = "NNKFUSED";
inline constexpr std::uint32_t kKernelCacheVersion = 1;
inline constexpr const char* kKernelCacheTrailer = "NNEND";
inline constexpr std::size_t kKernelCacheTrailerLen = 5;
inline constexpr std::size_t kKernelCacheShaLen = 32;

// 运行期加载形态的融合 kernel（自有存储；区别于旧的编译期内嵌指针形态）。
struct FusedKernel
{
    std::string key;
    ExprSpec spec;
    std::vector<std::uint32_t> spirv;
    int reduce_axis = -1;          // -1 逐元素 / 0 行归约 / 1 列归约
    int has_matmul = 0;
    std::uint32_t view_param_count = 0;
    std::uint32_t rparam_count = 0;
    std::uint32_t vec_width = 1;
    std::uint32_t prec_sig = 0;    // V0=0；V1=EXPR_PREC_SIG_DISPATCH；V2 位图
};

// ── 字节级序列化（little-endian，长度前缀）──────────────────────────────
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
    std::size_t end_pos() const { return pos; }
};

// 标量容器写/读（float 向量：consts / rparams / fold.inits）
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

// ExprOperand / ExprView / ExprInstr / MatmulSpec / VecAccSpec
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
    return r.u8(v.kind) && r.u8(v.negate_first_half) && r.u32(v.param) &&
           r.u32(v.param2);
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
    return r.u8(in.op) && r.u8(in.dst) && read_operand(r, in.a) &&
           read_operand(r, in.b) && read_operand(r, in.c);
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
    return r.u8(m.a_input) && r.u8(m.b_input) && r.u8(m.transA) &&
           r.u8(m.transB) && r.u32(m.k) && r.u32(m.batch);
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
    return r.u8(v.vec_state) && r.u8(v.weight_reg) && r.u8(v.b_input) &&
           r.u8(v.scale_reg) && r.u8(v.has_scale);
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

// ExprSpec：instrs / views / consts / rparams / num_regs / matmul / fold / extras
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

// ── 单 kernel 条目编码/解码 ──────────────────────────────────────────────
inline bool encode_kernel(detail::ByteWriter& w, const FusedKernel& k)
{
    w.len_bytes(k.key.data(), k.key.size());
    w.u32(static_cast<std::uint32_t>(k.spirv.size()));
    for (const auto x : k.spirv) w.u32(x);
    w.u32(static_cast<std::uint32_t>(k.reduce_axis));
    w.u32(static_cast<std::uint32_t>(k.has_matmul));
    w.u32(k.view_param_count);
    w.u32(k.rparam_count);
    w.u32(k.vec_width);
    w.u32(k.prec_sig);
    // spec 最后写（便于未来把 spirv/meta 前的 spec 挪成"延后解析"；顺序仅供阅读）
    const auto spec_start = w.buf.size();
    detail::write_expr_spec(w, k.spec);
    (void)spec_start;
    return true;
}

[[nodiscard]] inline bool decode_kernel(detail::ByteReader& r, FusedKernel& k)
{
    std::vector<std::uint8_t> key_bytes;
    if (!r.len_bytes(key_bytes)) return false;
    k.key.assign(reinterpret_cast<const char*>(key_bytes.data()), key_bytes.size());
    std::uint32_t nwords;
    if (!r.u32(nwords)) return false;
    k.spirv.resize(nwords);
    for (auto& x : k.spirv)
        if (!r.u32(x)) return false;
    std::uint32_t u;
    if (!r.u32(u)) return false; k.reduce_axis = static_cast<int>(u);
    if (!r.u32(u)) return false; k.has_matmul = static_cast<int>(u);
    if (!r.u32(k.view_param_count)) return false;
    if (!r.u32(k.rparam_count)) return false;
    if (!r.u32(k.vec_width)) return false;
    if (!r.u32(k.prec_sig)) return false;
    return detail::read_expr_spec(r, k.spec);
}

// ── 整文件 .nnk 编码/解码（含 magic/version/count/sha256/trailer）────────
[[nodiscard]] inline std::vector<std::uint8_t> encode_kernel_cache(
    const std::vector<FusedKernel>& kernels)
{
    detail::ByteWriter w;
    w.raw_bytes(kKernelCacheMagic, std::strlen(kKernelCacheMagic));
    w.u32(kKernelCacheVersion);
    w.u32(1u);  // flags：仅 GPU 融合算子
    w.u32(static_cast<std::uint32_t>(kernels.size()));
    for (const auto& k : kernels) encode_kernel(w, k);
    const auto body_len = w.buf.size();
    const auto digest = nn::detail::sha256_digest(
        std::string_view(reinterpret_cast<const char*>(w.buf.data()), body_len));
    w.raw_bytes(digest.data(), digest.size());
    w.raw_bytes(kKernelCacheTrailer, kKernelCacheTrailerLen);
    return w.buf;
}

// 解码并校验整文件 .nnk；返回空 vector 且 err = 具体错误信息。
[[nodiscard]] inline Result<std::vector<FusedKernel>> decode_kernel_cache(
    const std::uint8_t* data, std::size_t n)
{
    detail::ByteReader r{data, n, 0, false};
    constexpr std::size_t kMagicLen = 8;  // "NNKFUSED"
    if (!r.need(kMagicLen))
        NN_FAIL("kernel_cache: 文件过短/损坏（缺 magic）");
    if (std::memcmp(data, kKernelCacheMagic, kMagicLen) != 0)
        NN_FAIL("kernel_cache: magic 不符（非 kernels.nnk）");
    r.pos = kMagicLen;
    std::uint32_t version;
    if (!r.u32(version)) NN_FAIL("kernel_cache: 损坏（缺 version）");
    if (version != kKernelCacheVersion)
        NN_FAIL("kernel_cache: 版本不兼容（期望 " + std::to_string(kKernelCacheVersion) +
                "，实际 " + std::to_string(version) + "）");
    std::uint32_t flags, count;
    if (!r.u32(flags) || !r.u32(count)) NN_FAIL("kernel_cache: 损坏（缺 flags/count）");
    if (r.pos + 32 + kKernelCacheTrailerLen > n)
        NN_FAIL("kernel_cache: 文件过短（缺 sha256/trailer）");
    std::vector<FusedKernel> out;
    out.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        FusedKernel k;
        if (!decode_kernel(r, k)) NN_FAIL("kernel_cache: 条目 " + std::to_string(i) + " 损坏");
        out.push_back(std::move(k));
    }
    const auto body_len = r.end_pos();
    if (body_len + kKernelCacheShaLen + kKernelCacheTrailerLen != n)
        NN_FAIL("kernel_cache: 文件长度与条目不符");
    const auto digest = nn::detail::sha256_digest(
        std::string_view(reinterpret_cast<const char*>(data), body_len));
    if (std::memcmp(data + body_len, digest.data(), kKernelCacheShaLen) != 0)
        NN_FAIL("kernel_cache: 校验和不匹配（文件损坏或截断）");
    if (std::memcmp(data + body_len + kKernelCacheShaLen, kKernelCacheTrailer,
                    kKernelCacheTrailerLen) != 0)
        NN_FAIL("kernel_cache: trailer 不符");
    return out;
}

// ── 加载器（引擎运行期用）───────────────────────────────────────────────
struct KernelCache
{
    std::vector<FusedKernel> kernels;
    std::unordered_map<std::string, std::size_t> index;

    [[nodiscard]] static Result<KernelCache> from_bytes(const std::uint8_t* data,
                                                        std::size_t n)
    {
        auto kernels_r = decode_kernel_cache(data, n);
        if (!kernels_r) return std::unexpected(kernels_r.error());
        KernelCache c;
        c.kernels = std::move(*kernels_r);
        for (std::size_t i = 0; i < c.kernels.size(); ++i)
            c.index.emplace(c.kernels[i].key, i);
        return c;
    }

    [[nodiscard]] const FusedKernel* hit(const std::string& key) const noexcept
    {
        const auto it = index.find(key);
        if (it == index.end()) return nullptr;
        return &kernels[it->second];
    }
    [[nodiscard]] bool contains(const std::string& key) const noexcept
    { return index.find(key) != index.end(); }
};

} // namespace nn::fused
