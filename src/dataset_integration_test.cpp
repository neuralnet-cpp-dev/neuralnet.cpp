// ── dataset_integration_test — 分词器 × 数据集端到端集成 ────────────────────
// 本测试覆盖「真实 BPE 训练 → 并行 tokenize → assistant 掩码 → 写读
// .nndataset 逐位对拍」的集成链路（原 neuralnet.cpp 主仓 dataset_test 的
// 分词器耦合段）。纯格式测试在 dataset 仓的 dataset_test。
//
// 覆盖：
//   1. 并行 tokenize 保序 = 顺序 encode（任意并行度逐字节一致）
//   2. 逐 doc 掩码 = 全局 mark_assistant_spans（doc 边界重置）逐位一致
//   3. BPE 训练产物写 .nndataset → Dataset::load_text 逐位一致
//      （token_flow / doc_ids / loss_mask；空 token doc 不丢行号）
//   4. 词表 kvrec 往返稳定（模型内嵌词表字节来源）
//   5. .nnvocab 配套 .nndataset 的独立加载（save → Dataset::open 嗅探）
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
import nn.tokenizer;
#else
#include <neuralnet.cpp/dataset.hpp>
#include <neuralnet.cpp/domain_tokenizer.hpp>
#endif

int g_fail = 0;

#include "test_common.hpp"

namespace
{

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

// 对话语料：含 <|user|>/<|assistant|> 标记与中英文混合行
const std::string kCorpus =
    "the quick brown fox jumps over the lazy dog\r\n"
    "   padded line with spaces   \n"
    "\n"
    "\t tab padded \t\r\n"
    "<|user|>what is machine learning<|end_of_user|><|assistant|>it is a field of "
    "study<|end_of_assistant|>\n"
    "<|user|>hello there<|end_of_user|><|assistant|>hi friend<|end_of_assistant|>\n"
    "人工神经网络深度学习训练中。\n"
    "<|user|>unclosed assistant doc<|end_of_user|><|assistant|>tail without end marker\n"
    "final line of corpus\n";

}  // namespace

int main()
{
    std::printf("dataset_integration_test:\n");
    const std::string corpus_path = "dataset_integration_corpus.txt";
    write_file(corpus_path, kCorpus);

    // 读 docs（行 trim 规则在 dataset 仓；此处只用产物）
    auto docs_r = nn::read_text_docs(corpus_path);
    CHECK(docs_r.has_value(), "read_text_docs 成功");
    if (!docs_r) return 1;
    const std::vector<std::string>& docs = *docs_r;
    CHECK(docs.size() == 8, "doc 数正确（空行被跳过）");

    // ── BPE 训练（真实分词器）────────────────────────────────────
    nn::BPETokenizer tok;
    nn::BPETokenizer::Config tcfg;
    tcfg.vocab_size = 320;
    tcfg.min_freq = 1;
    tcfg.log = [](std::string_view) {};
    auto tr = tok.train(kCorpus, tcfg);
    CHECK(tr.has_value(), "分词器训练成功");
    if (!tr) return 1;

    // ── 1. 并行 tokenize 保序 = 顺序 encode ─────────────────────
    const auto per_doc = nn::encode_docs_parallel(tok, docs);
    bool same = per_doc.size() == docs.size();
    std::vector<std::size_t> ref_flow, ref_doc_ids;
    for (std::size_t i = 0; same && i < docs.size(); ++i)
    {
        const auto ref_toks = tok.encode(docs[i]);
        if (per_doc[i] != ref_toks) same = false;
        for (std::size_t t : ref_toks)
        {
            ref_flow.push_back(t);
            ref_doc_ids.push_back(i + 1);
        }
    }
    CHECK(same, "并行 tokenize 与顺序 encode 逐字节一致");

    // ── 2. 逐 doc 掩码 = 全局扫描参考逐位一致 ───────────────────
    std::vector<unsigned char> ref_mask_flat;
    {
        std::size_t asst_count = 0;
        ref_mask_flat = ref_mark_assistant_spans(ref_flow, ref_doc_ids, tok.assistant_marker_id(),
                                                 tok.end_assistant_marker_id(), asst_count);
        CHECK(asst_count > 0, "参考掩码含 assistant 目标");
    }
    std::vector<unsigned char> per_doc_mask_flat;
    std::vector<std::vector<unsigned char>> per_doc_masks;
    for (const auto& toks : per_doc)
    {
        per_doc_masks.push_back(nn::mark_assistant_span(toks, tok.assistant_marker_id(),
                                                        tok.end_assistant_marker_id()));
        per_doc_mask_flat.insert(per_doc_mask_flat.end(), per_doc_masks.back().begin(),
                                 per_doc_masks.back().end());
    }
    CHECK(per_doc_mask_flat == ref_mask_flat, "逐 doc 掩码与全局 mark_assistant_spans 逐位一致");

    // ── 3/4. 写 .nndataset → Dataset 读回逐位一致 ──────────────
    for (int mode = 0; mode < 2; ++mode)
    {
        const bool assistant = (mode == 1);
        const std::string tag = assistant ? "assistant" : "all";
        nn::NndatasetBuild build;
        build.vocab_kvrec = tok.to_kvrec();
        build.source_path = corpus_path;
        build.source_sha256 = nn::sha256_hex(kCorpus);
        build.source_docs = docs.size();
        build.loss_scope = assistant ? "assistant" : "all";
        build.tool = "dataset_gen v1";
        build.created = 12345;
        build.train_docs = per_doc;
        build.train_docs.insert(build.train_docs.begin() + 3, std::vector<std::size_t>{});
        if (assistant)
        {
            build.train_masks = per_doc_masks;
            build.train_masks.insert(build.train_masks.begin() + 3, std::vector<unsigned char>{});
        }
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
        for (std::size_t i = 0; i < 3; ++i)
        {
            build.test_docs.push_back(per_doc[i]);
            if (assistant) build.test_masks.push_back(per_doc_masks[i]);
        }

        const std::string path = "dataset_integration_" + tag + ".nndataset";
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
        CHECK(ds.info().loss_scope == build.loss_scope, (tag + ": loss_scope 一致").c_str());

        // 词表 kvrec 往返稳定（模型内嵌词表字节来源）
        CHECK(nn::write_nnvocab(ds.vocab_kvrec()) == nn::write_nnvocab(build.vocab_kvrec),
              (tag + ": 词表 kvrec 往返字节稳定").c_str());

        auto corpus_rr = ds.load_text(nn::DatasetSubset::Train);
        CHECK(corpus_rr.has_value(), (tag + ": load_text(train) 成功").c_str());
        if (!corpus_rr) continue;
        CHECK(corpus_rr->token_flow == exp_flow, (tag + ": token_flow 逐位一致").c_str());
        CHECK(corpus_rr->doc_ids == exp_ids,
              (tag + ": doc_ids 逐位一致（空 doc 不丢行号）").c_str());
        CHECK(corpus_rr->loss_mask == exp_mask,
              (tag + (assistant ? ": loss_mask 逐位一致" : ": loss_scope=all 合成全 1 掩码"))
                  .c_str());

        auto test_rr = ds.load_text(nn::DatasetSubset::Test);
        CHECK(test_rr.has_value(), (tag + ": load_text(test) 成功").c_str());
        if (test_rr)
        {
            std::vector<std::size_t> t_flow, t_ids;
            std::vector<unsigned char> t_mask;
            for (std::size_t i = 0; i < 3; ++i)
            {
                for (std::size_t t : per_doc[i])
                {
                    t_flow.push_back(t);
                    t_ids.push_back(i + 1);
                }
                if (assistant)
                    t_mask.insert(t_mask.end(), per_doc_masks[i].begin(), per_doc_masks[i].end());
                else t_mask.insert(t_mask.end(), per_doc[i].size(), 1u);
            }
            CHECK(test_rr->token_flow == t_flow && test_rr->doc_ids == t_ids &&
                      test_rr->loss_mask == t_mask,
                  (tag + ": test 子集逐位一致").c_str());
        }
    }

    // ── 5. .nnvocab 独立打开（词表独立加载场景）──────────────────
    {
        const std::string vpath = "dataset_integration_vocab.nnvocab";
        auto sv = tok.save(vpath);
        CHECK(sv.has_value(), "保存 .nnvocab");
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

    std::error_code ec;
    for (const char* p :
         {"dataset_integration_corpus.txt", "dataset_integration_all.nndataset",
          "dataset_integration_assistant.nndataset", "dataset_integration_vocab.nnvocab"})
        std::filesystem::remove(p, ec);

    std::printf("\ndataset_integration_test: %d failure(s)\n", g_fail);
    return g_fail != 0 ? 1 : 0;
}
