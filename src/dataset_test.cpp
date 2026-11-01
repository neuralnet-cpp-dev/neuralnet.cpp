// ── dataset_test — .nndataset / .nnvocab（nn::Dataset）纯格式测试 ────────────
// 本测试**不经分词器**（token 流手工构造）：dataset 仓的依赖面只有
// base + kvrecord。BPE 训练 → 并行 tokenize → 掩码 → 写读 .nndataset 的
// 端到端集成链路在 tokenizer 仓的 dataset_integration_test。
//
// 覆盖：
//   1. read_text_docs 行 trim 规则（与只读参考实现逐字一致；空行/全空白跳过）
//   2. mark_assistant_span 掩码（与全局扫描参考逐位一致；doc 边界语义）
//   3. write_nndataset → Dataset::open → load_text 逐位 round-trip
//      （loss_scope=all / assistant 双模式；空 token doc 不丢行号；
//       train/test 双子集；配套元数据 sha256/loss_scope）
//   4. text_doc 按 doc 随机访问（doc_id / tokens / mask）
//   5. .nnvocab 独立打开：词表 kvrec 逐字节稳定、load_text 明确报错
//   6. 错误路径：坏 magic / 坏版本 / 缺失文件 / masks 长度不一致
// 退出码：0 = 通过，1 = 失败。
// ───────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifdef NN_USE_MODULES
import nn.dataset;
#else
#include <neuralnet.cpp/dataset.hpp>
#endif

int g_fail = 0;

#include "test_common.hpp"

namespace
{

// ── 原 text_train read_file_lines 的 trim 语义（只读参考实现）──────────
std::vector<std::string> ref_read_docs(const std::string& buffer)
{
    std::vector<std::string> docs;
    std::size_t pos = 0;
    const std::size_t total = buffer.size();
    while (pos < total)
    {
        std::size_t line_start = pos;
        while (pos < total && buffer[pos] != '\n')
            ++pos;
        std::size_t line_end = pos;
        if (pos < total) ++pos;
        while (line_start < line_end && (buffer[line_start] == ' ' || buffer[line_start] == '\t' ||
                                         buffer[line_start] == '\r'))
            ++line_start;
        while (line_end > line_start &&
               (buffer[line_end - 1] == ' ' || buffer[line_end - 1] == '\t' ||
                buffer[line_end - 1] == '\r'))
            --line_end;
        if (line_start < line_end) docs.emplace_back(buffer, line_start, line_end - line_start);
    }
    return docs;
}

// ── 原 text_train mark_assistant_spans（全局扫描 + doc 边界重置）────────
std::vector<unsigned char> ref_mark_assistant_spans(const std::vector<std::size_t>& flow,
                                                    const std::vector<std::size_t>& doc_ids,
                                                    std::size_t asst_begin, std::size_t asst_end,
                                                    std::size_t& asst_count)
{
    std::vector<unsigned char> mask(flow.size(), 0);
    bool in_asst = false;
    asst_count = 0;
    for (std::size_t i = 0; i < flow.size(); ++i)
    {
        if (!doc_ids.empty() && i > 0 && doc_ids[i] != doc_ids[i - 1]) in_asst = false;
        const std::size_t id = flow[i];
        if (id == asst_begin) in_asst = true;
        mask[i] = in_asst ? 1u : 0u;
        if (mask[i]) ++asst_count;
        if (id == asst_end) in_asst = false;
    }
    return mask;
}

void write_file(const std::string& path, const std::string& content)
{
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(content.data(), static_cast<std::streamsize>(content.size()));
}

std::string read_file(const std::string& path)
{
    std::ifstream ifs(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

const std::string kCorpus = "the quick brown fox jumps over the lazy dog\r\n"
                            "   padded line with spaces   \n"
                            "\n"
                            "\t tab padded \t\r\n"
                            "人工神经网络深度学习训练中。\n"
                            "final line of corpus\n";

// 假想对话标记 ID（掩码语义测试用；不依赖真实分词器）
constexpr std::size_t kAsstBegin = 101;
constexpr std::size_t kAsstEnd = 102;

// 手工构造的词表 kvrec（dataset 层不解释词表内容，只要求字节稳定）
nn::KeyValueRecord make_vocab()
{
    nn::KeyValueRecord vocab;
    vocab.set("kind", std::string{"bpe"});
    vocab.set("vocab_size", std::uint64_t{320});
    vocab.set("ids", std::vector<std::uint64_t>{0, 1, 2, 3, kAsstBegin, kAsstEnd});
    return vocab;
}

}  // namespace

int main()
{
    std::printf("dataset_test:\n");
    const std::string corpus_path = "dataset_test_corpus.txt";
    write_file(corpus_path, kCorpus);

    // ── 1. 行 trim 规则逐字一致 ─────────────────────────────────
    auto docs_r = nn::read_text_docs(corpus_path);
    CHECK(docs_r.has_value(), "read_text_docs 成功");
    if (!docs_r) return 1;
    const std::vector<std::string>& docs = *docs_r;
    CHECK(docs == ref_read_docs(kCorpus), "行 trim 规则与原 read_file_lines 逐字一致");
    CHECK(docs.size() == 5, "doc 数正确（空行/全空白行被跳过）");

    // ── 2. mark_assistant_span 掩码 = 全局扫描参考逐位一致 ─────
    // 手工 token 流：5 个 doc（第 3 个为空 doc），doc 内含 assistant 段
    std::vector<std::vector<std::size_t>> tokens{
        {5, kAsstBegin, 6, 7, kAsstEnd, 8},          // doc 1：一个 assistant 段
        {9, 10, 11},                                  // doc 2：无 assistant
        {},                                           // doc 3：空 doc
        {kAsstBegin, 12, kAsstEnd, kAsstBegin, 13, kAsstEnd},  // doc 4：两段
        {14, kAsstBegin, 15, kAsstEnd},               // doc 5：单段闭合
    };
    {
        std::vector<std::size_t> flow, doc_ids;
        std::vector<std::vector<unsigned char>> per_doc_masks;
        for (std::size_t i = 0; i < tokens.size(); ++i)
        {
            per_doc_masks.push_back(nn::mark_assistant_span(tokens[i], kAsstBegin, kAsstEnd));
            for (std::size_t t : tokens[i])
            {
                flow.push_back(t);
                doc_ids.push_back(i + 1);
            }
        }
        std::size_t asst_count = 0;
        const auto ref = ref_mark_assistant_spans(flow, doc_ids, kAsstBegin, kAsstEnd, asst_count);
        CHECK(asst_count > 0, "参考掩码含 assistant 目标");
        std::vector<unsigned char> flat;
        for (const auto& m : per_doc_masks)
            flat.insert(flat.end(), m.begin(), m.end());
        CHECK(flat == ref, "逐 doc 掩码与全局 mark_assistant_spans 逐位一致");
    }

    // ── 3. 写 .nndataset → 读回逐位一致（all / assistant 双模式）──
    for (int mode = 0; mode < 2; ++mode)
    {
        const bool assistant = (mode == 1);
        const std::string tag = assistant ? "assistant" : "all";
        nn::NndatasetBuild build;
        build.vocab_kvrec = make_vocab();
        build.source_path = corpus_path;
        build.source_sha256 = nn::sha256_hex(kCorpus);
        build.source_docs = docs.size();
        build.loss_scope = assistant ? "assistant" : "all";
        build.tool = "dataset_test";
        build.created = 12345;

        // train：插一个空 token doc（不丢行号语义）
        build.train_docs = tokens;
        build.train_docs.insert(build.train_docs.begin() + 3, std::vector<std::size_t>{});
        std::vector<std::vector<unsigned char>> train_masks;
        if (assistant)
        {
            for (const auto& t : tokens)
                train_masks.push_back(nn::mark_assistant_span(t, kAsstBegin, kAsstEnd));
            train_masks.insert(train_masks.begin() + 3, std::vector<unsigned char>{});
            build.train_masks = std::move(train_masks);
        }
        // test：前 3 个 doc
        for (std::size_t i = 0; i < 3; ++i)
        {
            build.test_docs.push_back(tokens[i]);
            if (assistant)
                build.test_masks.push_back(
                    nn::mark_assistant_span(tokens[i], kAsstBegin, kAsstEnd));
        }

        // 期望产物
        std::vector<std::size_t> exp_flow, exp_ids;
        std::vector<unsigned char> exp_mask;
        for (std::size_t i = 0; i < build.train_docs.size(); ++i)
        {
            for (std::size_t t : build.train_docs[i])
            {
                exp_flow.push_back(t);
                exp_ids.push_back(i + 1);
            }
            if (assistant)
                exp_mask.insert(exp_mask.end(), build.train_masks[i].begin(),
                                build.train_masks[i].end());
            else exp_mask.insert(exp_mask.end(), build.train_docs[i].size(), 1u);
        }

        const std::string path = "dataset_test_" + tag + ".nndataset";
        {
            std::ostringstream oss;
            auto wr = nn::write_nndataset(oss, build);
            CHECK(wr.has_value(), (tag + ": write_nndataset 成功").c_str());
            if (!wr) continue;
            write_file(path, nn::write_file_label(nn::NNDATASET_MAGIC, nn::NNDATASET_FILE_VERSION) +
                                 oss.str());
        }

        auto ds_r = nn::Dataset::open(path);
        CHECK(ds_r.has_value(), (tag + ": Dataset::open 成功").c_str());
        if (!ds_r) continue;
        const nn::Dataset& ds = *ds_r;
        CHECK(ds.info().kind == nn::DatasetKind::Text, (tag + ": kind=text").c_str());
        CHECK(ds.info().num_docs == build.train_docs.size(),
              (tag + ": num_docs（含空 doc）").c_str());
        CHECK(ds.info().test_docs == 3, (tag + ": test 子集元数据").c_str());
        CHECK(ds.info().loss_scope == build.loss_scope, (tag + ": loss_scope 一致").c_str());
        CHECK(ds.info().source_sha256 == nn::sha256_hex(kCorpus),
              (tag + ": source.sha256 一致").c_str());

        // 词表 kvrec 往返稳定
        CHECK(nn::write_nnvocab(ds.vocab_kvrec()) == nn::write_nnvocab(build.vocab_kvrec),
              (tag + ": 词表 kvrec 往返字节稳定").c_str());

        // train 子集逐位一致（空 doc 不产生 token、但不改后续 doc_id）
        auto corpus_r = ds.load_text(nn::DatasetSubset::Train);
        CHECK(corpus_r.has_value(), (tag + ": load_text(train) 成功").c_str());
        if (!corpus_r) continue;
        CHECK(corpus_r->token_flow == exp_flow, (tag + ": token_flow 逐位一致").c_str());
        CHECK(corpus_r->doc_ids == exp_ids,
              (tag + ": doc_ids 逐位一致（空 doc 不丢行号）").c_str());
        CHECK(corpus_r->loss_mask == exp_mask,
              (tag + (assistant ? ": loss_mask 逐位一致" : ": loss_scope=all 合成全 1 掩码"))
                  .c_str());

        // test 子集（前 3 个 doc）逐位一致
        auto test_r = ds.load_text(nn::DatasetSubset::Test);
        CHECK(test_r.has_value(), (tag + ": load_text(test) 成功").c_str());
        if (test_r)
        {
            std::vector<std::size_t> t_flow, t_ids;
            std::vector<unsigned char> t_mask;
            for (std::size_t i = 0; i < 3; ++i)
            {
                for (std::size_t t : tokens[i])
                {
                    t_flow.push_back(t);
                    t_ids.push_back(i + 1);
                }
                if (assistant)
                {
                    const auto m = nn::mark_assistant_span(tokens[i], kAsstBegin, kAsstEnd);
                    t_mask.insert(t_mask.end(), m.begin(), m.end());
                }
                else t_mask.insert(t_mask.end(), tokens[i].size(), 1u);
            }
            CHECK(test_r->token_flow == t_flow && test_r->doc_ids == t_ids &&
                      test_r->loss_mask == t_mask,
                  (tag + ": test 子集逐位一致").c_str());
        }

        // ── 4. text_doc 随机访问 ────────────────────────────────
        {
            auto d0 = ds.text_doc(nn::DatasetSubset::Train, 0);
            CHECK(d0.has_value() && d0->doc_id == 1 && d0->tokens == build.train_docs[0],
                  (tag + ": text_doc(train,0) 内容一致").c_str());
            // 空 doc（下标 3）
            auto d3 = ds.text_doc(nn::DatasetSubset::Train, 3);
            CHECK(d3.has_value() && d3->tokens.empty(), (tag + ": text_doc 空 doc").c_str());
            auto bad = ds.text_doc(nn::DatasetSubset::Train, build.train_docs.size() + 10);
            CHECK(!bad.has_value(), (tag + ": text_doc 越界报错").c_str());
        }
    }

    // ── 5. .nnvocab 独立打开 ─────────────────────────────────────
    {
        const std::string vpath = "dataset_test_vocab.nnvocab";
        write_file(vpath, nn::write_nnvocab(make_vocab()));
        auto ds_r = nn::Dataset::open(vpath);
        CHECK(ds_r.has_value(), ".nnvocab 可被 Dataset::open 嗅探打开");
        if (ds_r)
        {
            auto bad = ds_r->load_text(nn::DatasetSubset::Train);
            CHECK(!bad.has_value(), ".nnvocab 的 load_text 明确报错（不含数据集子集）");
            CHECK(nn::write_nnvocab(ds_r->vocab_kvrec()) == read_file(vpath),
                  ".nnvocab 词表 kvrec 逐字节稳定");
        }
    }

    // ── 6. 错误路径 ──────────────────────────────────────────────
    {
        auto missing = nn::Dataset::open("dataset_test_no_such_file.nndataset");
        CHECK(!missing.has_value(), "缺失文件报错");

        write_file("dataset_test_bad.bin", "NOTNNDSxxxxxxxx");
        auto bad_magic = nn::Dataset::open("dataset_test_bad.bin");
        CHECK(!bad_magic.has_value(), "坏 magic 报错");

        // 好 magic + 坏版本
        std::string bad_ver = nn::write_file_label(nn::NNDATASET_MAGIC, 9999);
        bad_ver += std::string(64, '\0');
        write_file("dataset_test_badver.bin", bad_ver);
        auto bad_v = nn::Dataset::open("dataset_test_badver.bin");
        CHECK(!bad_v.has_value(), "不支持的文件版本报错");

        // assistant 模式 masks 长度不一致
        nn::NndatasetBuild b;
        b.vocab_kvrec = make_vocab();
        b.loss_scope = "assistant";
        b.train_docs = {{1, 2, 3}};
        b.train_masks = {{1, 0}};   // 长度 2 != 3
        std::ostringstream oss;
        auto wr = nn::write_nndataset(oss, b);
        CHECK(!wr.has_value(), "掩码长度不一致 → write_nndataset 报错");

        // all 模式不允许携带 masks
        nn::NndatasetBuild b2;
        b2.vocab_kvrec = make_vocab();
        b2.loss_scope = "all";
        b2.train_docs = {{1, 2, 3}};
        b2.train_masks = {{1, 1, 1}};
        std::ostringstream oss2;
        auto wr2 = nn::write_nndataset(oss2, b2);
        CHECK(!wr2.has_value(), "all 模式携带 masks → 报错");
    }

    // 清理
    std::error_code ec;
    for (const char* p : {"dataset_test_corpus.txt", "dataset_test_all.nndataset",
                          "dataset_test_assistant.nndataset", "dataset_test_vocab.nnvocab",
                          "dataset_test_bad.bin", "dataset_test_badver.bin"})
        std::filesystem::remove(p, ec);

    std::printf("\ndataset_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
