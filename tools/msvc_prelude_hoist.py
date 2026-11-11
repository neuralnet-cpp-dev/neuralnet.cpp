# 聚合体 prelude 提升（一次性整改 + 可重跑幂等）
# MSVC 模块消费方对「先 import 后文本 include 同一头」不合并实体（C2953/C2572 族，
# 上游 microsoft/STL#6436 同类）。聚合体 flatten 里子文件 A 的 import 会先于子文件
# B 的 include ⇒ 在首个嵌套 .cpp 之前补入 flatten 全部头（include guard 幂等，
# clang/GCC 下无害）。幂等：已在插入点上方出现的行跳过；重复运行 prelude 已存在则
# 以「上方集合」去重跳过。
import re
import sys
from pathlib import Path

AGGS = [
    "attn_test.cpp", "cnn_test.cpp", "doc_attn_test.cpp", "expr_cpu_test.cpp",
    "expr_gpu_test.cpp", "gpt_test.cpp", "layer_gradcheck_test.cpp",
    "precision_test.cpp", "rapt_test.cpp", "tokenizer_test.cpp",
]
ROOT = Path(__file__).resolve().parent.parent / "src"
INC_CPP = re.compile(r'^\s*#include\s+"([^"]+\.cpp)"\s*$')
INC_ANY = re.compile(r'^\s*#include\s+(<[^>]+>|"([^"]+)")\s*$')


def flatten_includes(entry: str):
    seen, queue, out = set(), [entry], []
    while queue:
        f = queue.pop(0)
        if f in seen:
            continue
        seen.add(f)
        p = ROOT / f
        if not p.exists():
            print(f"  !! missing {f}", file=sys.stderr)
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            m = INC_ANY.match(line)
            if not m:
                continue
            if m.group(2) and m.group(2).endswith(".cpp"):
                queue.append(m.group(2))
            else:
                out.append(line.strip())
    return out


def main():
    for a in AGGS:
        path = ROOT / a
        raw = path.read_bytes()
        crlf = b"\r\n" in raw
        text = raw.decode("utf-8")
        lines = text.splitlines(keepends=True)

        idx = None
        for i, l in enumerate(lines):
            if INC_CPP.match(l):
                idx = i
                break
        if idx is None:
            print(f"{a}: NO-NESTED-CPP (skip)")
            continue

        # 幂等：若 prelude 标记已存在，先不重复插入（上方集合会吃掉全部行）
        uniq, s = [], set()
        for inc in flatten_includes(a):
            if inc not in s:
                s.add(inc)
                uniq.append(inc)

        above = {l.strip() for l in lines[:idx]}
        prelude = [u for u in uniq if u not in above]
        if not prelude:
            print(f"{a}: already-clean (0 lines to hoist)")
            continue

        eol = "\r\n" if crlf else "\n"
        block = [
            f"// ── prelude（聚合体头提升）───────────────────────────────────────{eol}",
            f"// 下方首个嵌套 .cpp 起即出现 import；MSVC 模块消费方对「import 之后再{eol}",
            f"// 文本 include 同一头」不合并实体（C2953/C2572 族，上游 microsoft/STL{eol}",
            f"// #6436 同类）。故把 flatten（本文件+全部嵌套 .cpp）的头在此先行包含；{eol}",
            f"// include guard 幂等，clang/GCC 路径无害。改动此块须同步嵌套文件的头集合。{eol}",
        ]
        block += [u + eol for u in prelude]
        block.append(eol)

        new_lines = lines[:idx] + block + lines[idx:]
        path.write_bytes("".join(new_lines).encode("utf-8"))
        print(f"{a}: hoisted {len(prelude)} includes (insert at line {idx + 1})")


if __name__ == "__main__":
    main()
