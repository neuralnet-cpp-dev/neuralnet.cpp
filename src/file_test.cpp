// ── file_test — core_file.hpp：文件读写 + 流式摘要 + POD I/O ────────────────
// 覆盖：
//   1. load_text_file：读全文件（二进制模式，保留 \r\n）/ 不存在文件报错
//   2. sha256_file_prefix / sha256_file_hex：整文件摘要 = 内容 sha256_hex；
//      limit_bytes 前缀摘要 = 前缀内容摘要；短于 limit_bytes 报错
//   3. write_pod / read_pod / write_pod_span / read_pod_span 往返
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <neuralnet.cpp/core_file.hpp>

int g_fail = 0;

#include "test_common.hpp"

namespace
{

const char* kPath = "file_test_tmp.bin";    // 测试临时文件（结束时删除）

void write_all(const std::string& path, const std::string& bytes)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void test_load_text_file()
{
    std::printf("load_text_file:\n");
    const std::string content = "line1\r\nline2\n\ttab 中文\n";
    write_all(kPath, content);

    auto r = nn::load_text_file(kPath);
    CHECK(r.has_value(), "读取成功");
    if (r) CHECK(*r == content, "内容逐字节一致（含 \\r\\n 与 UTF-8）");

    auto bad = nn::load_text_file("file_test_missing_no_such_file.txt");
    CHECK(!bad.has_value(), "不存在的文件报错");
    if (!bad)
        CHECK(bad.error().message.find("file_test_missing_no_such_file.txt") != std::string::npos,
              "错误消息含路径");
}

void test_sha256_file()
{
    std::printf("sha256_file_prefix / sha256_file_hex:\n");
    const std::string content = "the quick brown fox jumps over the lazy dog\n";
    write_all(kPath, content);
    const std::string ref = nn::sha256_hex(content);

    auto whole = nn::sha256_file_hex(kPath);
    CHECK(whole.has_value(), "整文件摘要成功");
    if (whole) CHECK(*whole == ref, "整文件摘要 = 内容 sha256_hex");

    // limit_bytes 前缀摘要 = 前缀内容摘要
    for (std::uint64_t n :
         {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{10}, std::uint64_t{content.size()}})
    {
        auto p = nn::sha256_file_prefix(kPath, n);
        CHECK(p.has_value(), "前缀摘要成功");
        if (p)
            CHECK(nn::sha256_hex(*p) == nn::sha256_hex(std::string_view(content).substr(
                                            0, static_cast<std::size_t>(n))),
                  ("前缀 " + std::to_string(n) + " 字节").c_str());
    }

    // 短于 limit_bytes → 报错
    auto short_r = nn::sha256_file_prefix(kPath, content.size() + 1);
    CHECK(!short_r.has_value(), "文件短于摘要范围报错");

    // 不存在的文件
    auto missing = nn::sha256_file_hex("file_test_missing_no_such_file.bin");
    CHECK(!missing.has_value(), "不存在的文件摘要报错");
}

void test_pod_io()
{
    std::printf("write_pod / read_pod / span 往返:\n");
    // 单对象
    {
        std::ofstream ofs(kPath, std::ios::binary);
        const std::uint64_t v = 0x0123456789ABCDEFull;
        CHECK(nn::write_pod(ofs, v), "write_pod 成功");
    }
    {
        std::ifstream ifs(kPath, std::ios::binary);
        std::uint64_t v = 0;
        CHECK(nn::read_pod(ifs, v), "read_pod 成功");
        CHECK(v == 0x0123456789ABCDEFull, "值往返一致");
        std::uint64_t extra = 0;
        CHECK(!nn::read_pod(ifs, extra), "EOF 后 read_pod 失败");
    }
    // span（float 数组）
    constexpr std::size_t kN = 5;
    {
        const std::vector<float> src{0.0f, -1.5f, 2.25f, 1e30f, -0.0f};
        std::ofstream ofs(kPath, std::ios::binary);
        CHECK(nn::write_pod_span(ofs, std::span(src)), "write_pod_span 成功");
    }
    {
        std::vector<float> dst(kN);
        std::ifstream ifs(kPath, std::ios::binary);
        CHECK(nn::read_pod_span(ifs, std::span(dst)), "read_pod_span 成功");
        CHECK(dst.size() == kN && dst[1] == -1.5f && dst[3] == 1e30f, "span 值往返一致");
    }
    // 空 span（合法：0 元素）
    {
        std::ofstream ofs(kPath, std::ios::binary);
        const std::vector<std::uint32_t> empty;
        CHECK(nn::write_pod_span(ofs, std::span(empty)), "空 span 写出成功");
        CHECK(std::filesystem::file_size(kPath) == 0, "空 span = 0 字节");
    }
}

}  // namespace

int main()
{
    std::puts("=== file I/O (core_file.hpp) ===");
    test_load_text_file();
    test_sha256_file();
    test_pod_io();

    std::error_code ec;
    std::filesystem::remove(kPath, ec);

    std::printf("\nfile_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
