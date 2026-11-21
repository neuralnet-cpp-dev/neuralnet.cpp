#!/usr/bin/env python3
"""migrate_kvrec.py — 一次性：nn::KeyValueRecord → kvrec::，include 切 <kvrec/kvrecord.hpp>。

用法（主仓根目录）：
    python tools/migrate_kvrec.py [--dry-run]

只处理 ripgrep 已确认的引用面；跳过 docs/history、release-notes、exstl/kvrec、
正在迁移的本脚本与历史一次性工具。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 由 rg 确认的相对路径清单（源码 + 构建 + 当前文档；历史册除外）
FILES = [
    "CMakeLists.txt",
    "AGENTS.md",
    "docs/development/02-cpp-api-manual.md",
    "docs/development/10-kvrecord-format-spec.md",
    "docs/internal/19-unified-dataset.md",
    "docs/internal/35-modules-migration-implementation.md",
    "include/neuralnet.cpp/dataset.hpp",
    "include/neuralnet.cpp/dataset_format.hpp",
    "include/neuralnet.cpp/model_spec.hpp",
    "include/neuralnet.cpp/model_serialization.hpp",
    "include/neuralnet.cpp/rl_pipeline.hpp",
    "include/neuralnet.cpp/train_state.hpp",
    "modules/nn.kvrecord.cpp",
    "modules/nn.engine.cpp",
    "scripts/kvrecord.py",
    "src/dataset_test.cpp",
    "src/dataset_samples_test.cpp",
    "src/jit_demo.cpp",
    "src/kvrec_test.cpp",
    "src/model_spec_validation_test.cpp",
    "src/rl_output_test.cpp",
    "src/train_state_test.cpp",
    "engine/CMakeLists.txt",
    "engine/abstraction/include/neuralnet.cpp/kernel_cache.hpp",
    "engine/engine/include/neuralnet.cpp/compute_gpu_engine.hpp",
]

# 待删除（实体迁出）
DELETE = [
    "engine/abstraction/include/neuralnet.cpp/model_keyvalue_record.hpp",
]

INCLUDE_RE = re.compile(
    r"#include\s*<neuralnet\.cpp/model_keyvalue_record\.hpp>(?P<comment>[^\n]*)"
)

SYMBOL_SUBS = [
    ("nn::detail::append_u32", "kvrec::detail::append_u32"),
    ("nn::detail::append_u64", "kvrec::detail::append_u64"),
    ("nn::detail::take_u32", "kvrec::detail::take_u32"),
    ("nn::detail::take_u64", "kvrec::detail::take_u64"),
    ("nn::kvrec_memory_source", "kvrec::kvrec_memory_source"),
    ("nn::KeyValueRecord", "kvrec::KeyValueRecord"),
    ("nn::KvReadAt", "kvrec::KvReadAt"),
    ("${NN_ABST_INC}/model_keyvalue_record.hpp", "${NN_KVREC_INC}/kvrecord.hpp"),
    (
        "engine/abstraction/include/neuralnet.cpp/model_keyvalue_record.hpp",
        "kvrec/include/kvrec/kvrecord.hpp",
    ),
]


def transform(text: str) -> str:
    def repl_include(m: re.Match[str]) -> str:
        comment = m.group("comment").rstrip()
        # 保留语义注释，统一指向 kvrec
        if comment:
            # 例： // nn::KeyValueRecord（write_kv/read_kv）
            #   →  // kvrec::KeyValueRecord（write_kv/read_kv）
            comment = comment.replace("nn::", "kvrec::")
            comment = comment.replace("model_keyvalue_record", "kvrec/kvrecord")
            if "kvrec" not in comment and "KeyValueRecord" not in comment:
                comment = "  // kvrec"
            return "#include <kvrec/kvrecord.hpp>" + comment
        return "#include <kvrec/kvrecord.hpp>"

    text = INCLUDE_RE.sub(repl_include, text)
    for a, b in SYMBOL_SUBS:
        text = text.replace(a, b)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed: list[str] = []
    missing: list[str] = []
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"skip (encoding): {rel}", file=sys.stderr)
            continue
        new = transform(text)
        if new != text:
            changed.append(rel)
            if not args.dry_run:
                path.write_text(new, encoding="utf-8", newline="\n")

    for rel in DELETE:
        path = ROOT / rel
        if path.exists() and not args.dry_run:
            path.unlink()
            print(f"deleted: {rel}")
        elif path.exists():
            print(f"[dry-run] would delete: {rel}")

    print(f"{'[dry-run] ' if args.dry_run else ''}changed {len(changed)} files:")
    for c in changed:
        print(f"  {c}")
    if missing:
        print("missing:")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
