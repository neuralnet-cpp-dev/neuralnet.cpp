// ── sha256_test — core_sha256.hpp：SHA-256（FIPS 180-4）─────────────────────
// 覆盖：
//   1. FIPS 180-4 / NIST 标准测试向量（空串 / "abc" / 448-bit 两块消息 /
//      896-bit 长消息）
//   2. 增量层与一次性层对拍（任意分块大小，含跨 64 字节块边界的拼接）
//   3. sha256_hex 格式（64 小写 hex）与摘要字节一致性
//   4. 雪崩性抽查（单比特翻转 → 摘要大变）
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstdio>
#include <string>

#include <neuralnet.cpp/core_sha256.hpp>

int g_fail = 0;

#include "test_common.hpp"

namespace
{

// FIPS 180-4 官方向量（十六进制摘要）
struct Vector
{
    const char* input;
    const char* sha256;
};

constexpr Vector kVectors[] = {
    // 空串（NIST 示例）
    {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    // "abc"
    {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
    // "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"（448 bit，两块消息）
    {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
     "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    // "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
    //   "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"（896 bit）
    {"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
     "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
     "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"},
};

void test_fips_vectors()
{
    std::printf("FIPS 180-4 标准向量:\n");
    for (const auto& v : kVectors)
    {
        const std::string got = nn::sha256_hex(v.input);
        CHECK(got == v.sha256, v.input[0] ? v.input : "<empty>");
        // 摘要字节重载与 hex 重载一致
        const auto d = nn::detail::sha256_digest(v.input);
        CHECK(nn::sha256_hex(d) == v.sha256, "sha256_hex(digest) 一致");
    }
}

void test_incremental_matches_oneshot()
{
    std::printf("增量层 vs 一次性层（多种分块）:\n");
    // 构造 > 200 字节的输入（覆盖 3 个以上块 + 尾块）
    std::string data;
    for (int i = 0; i < 60; ++i)
        data += "0123456789abcdef";   // 960 字节
    const std::string ref = nn::sha256_hex(data);

    // 各种分块大小（1/3/63/64/65/127/1000），含跨块边界
    const std::size_t chunks[] = {1, 3, 63, 64, 65, 127, 1000};
    for (std::size_t cs : chunks)
    {
        nn::detail::Sha256State st;
        for (std::size_t pos = 0; pos < data.size(); pos += cs)
            nn::detail::sha256_update(st, std::string_view(data).substr(pos, cs));
        const std::string got = nn::sha256_hex(nn::detail::sha256_finalize(st));
        CHECK(got == ref, ("分块 " + std::to_string(cs)).c_str());
    }

    // 空 update（零长度串）不改变摘要
    {
        nn::detail::Sha256State st;
        nn::detail::sha256_update(st, "");
        nn::detail::sha256_update(st, data);
        nn::detail::sha256_update(st, "");
        const std::string got = nn::sha256_hex(nn::detail::sha256_finalize(st));
        CHECK(got == ref, "零长度 update 不影响结果");
    }
}

void test_hex_format()
{
    std::printf("hex 格式:\n");
    const std::string h = nn::sha256_hex("abc");
    CHECK(h.size() == 64, "64 字符");
    bool lower = true;
    for (char c : h)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) lower = false;
    CHECK(lower, "全小写 hex");
}

void test_avalanche()
{
    std::printf("雪崩性抽查:\n");
    std::string a(200, 'x');
    std::string b = a;
    b[137] = static_cast<char>(b[137] ^ 0x01);   // 单比特翻转
    const std::string ha = nn::sha256_hex(a);
    const std::string hb = nn::sha256_hex(b);
    CHECK(ha != hb, "单比特翻转 → 摘要不同");
    std::size_t diff = 0;
    for (std::size_t i = 0; i < ha.size(); ++i)
        if (ha[i] != hb[i]) ++diff;
    // 期望翻转 ~50% 的 hex 字符（64 个中的绝大多数）；下界取 25% 留足余量
    CHECK(diff >= 16, "差异充分扩散（>=16/64 hex 字符不同）");
}

}  // namespace

int main()
{
    std::puts("=== sha256 (core_sha256.hpp) ===");
    test_fips_vectors();
    test_incremental_matches_oneshot();
    test_hex_format();
    test_avalanche();
    std::printf("\nsha256_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
