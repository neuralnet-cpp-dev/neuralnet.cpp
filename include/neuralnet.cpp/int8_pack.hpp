#pragma once

// ── int8_pack.hpp — int8 激活的打包布局与编解码（W1.58A8，P3 显存侧）──────────
//
// 背景（docs/internal/21-quantized-weights.md §4.16 的 P3）：BitNet 训练配方把
// 线性投影的激活量化到 8-bit 整数（W1.58A8：三值权重 × int8 激活，per-token
// absmax 尺度）——训练期激活缓存从 f16 2 B/元素降到 **1 B/元素**，是训练显存
// 的大头削减（b1.58 论文"trained with 8-bit activations"的落地）。
//
// 布局（与 ternary_pack 同一先例：容器 = f32 张量承载原始字）：
//   · 8 bit / 元素（signed int8，值域 [-127,127]）；4 元素 / 32-bit 字；
//   · 行内小端：第 c 个元素的字节在 `(word >> 8·(c%4)) & 0xFF`；
//   · 形状 (rows, ceil(cols/4)) → 1 B/元素（f16 稠密 2 B、f32 4 B）；
//   · 尾部（idx ≥ cols）填 0。
//
// 量化公式（与 BitNet a2.6 口径一致，docs 21 §4.16 ③）：
//   q = round(clamp(x/s, -127, 127))     （round-half-away-from-zero；先 clamp
//      后 round ⇒ 结果恒在 [-127,127]，无溢出；NaN → 0）
//   s 按**列**（per-token absmax，激活）或按**行**（优化器状态/权重侧）广播，
//   由调用方（原语内核）决定 —— 本头只提供逐元素 helper。
// ─────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <cstdint>

namespace nn
{
namespace i8_pack
{

inline constexpr std::uint32_t kBits = 8;                 // 每元素 8 bit
inline constexpr std::uint32_t kPerWord = 32 / kBits;     // 4 元素 / f32 元素
inline constexpr std::uint32_t kMask = (1u << kBits) - 1u;

// 每行需要的字（f32 元素）数：(N + 3) / 4
[[nodiscard]] inline constexpr std::size_t words_per_row(std::size_t n) noexcept
{ return (n + kPerWord - 1) / kPerWord; }

// 量化（round-half-away-from-zero；先 clamp 后 round ⇒ 恒 ∈ [-127,127]）
// NaN → 0（确定性：不把 NaN 位型带进 int 转换）。
[[nodiscard]] inline std::int8_t quantize(float v, float s) noexcept
{
    const float scaled = v / s;
    // clamp 到 [-127,127]（NaN 比较全 false → 落到 else 分支置 0）
    float c;
    if (scaled >= 127.0f) c = 127.0f;
    else if (scaled <= -127.0f) c = -127.0f;
    else if (scaled != scaled) c = 0.0f;   // NaN
    else c = scaled;
    // round-half-away：c + 0.5·sign → trunc（向零）——与 std::round 语义一致，
    // 但用整数截断实现，不依赖宿主舍入模式（铁律 8 确定性）。
    const float r = c + (c < 0.0f ? -0.5f : 0.5f);
    return static_cast<std::int8_t>(static_cast<int>(r));
}

// 打包一行：q[0..N)（int8 宿主连续）→ words_per_row(N) 个 32-bit 字。
// 尾部填 0。
inline void pack_row(const std::int8_t* q, std::size_t n, std::uint32_t* out) noexcept
{
    const std::size_t words = words_per_row(n);
    for (std::size_t wi = 0; wi < words; ++wi)
    {
        std::uint32_t word = 0;
        for (std::uint32_t j = 0; j < kPerWord; ++j)
        {
            const std::size_t idx = wi * kPerWord + j;
            const std::uint32_t byte = (idx < n) ? static_cast<std::uint8_t>(q[idx]) : 0u;
            word |= byte << (kBits * j);
        }
        out[wi] = word;
    }
}

// 单点解码（signed int8）：words 指向该行首字
[[nodiscard]] inline std::int8_t at(const std::uint32_t* words, std::size_t idx) noexcept
{
    const std::uint32_t word = words[idx / kPerWord];
    const std::uint32_t byte = (word >> (kBits * (idx % kPerWord))) & kMask;
    return static_cast<std::int8_t>(static_cast<std::uint8_t>(byte));
}

} // namespace i8_pack
} // namespace nn
