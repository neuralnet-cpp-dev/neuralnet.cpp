# Qualify remaining bare foundation names inside nn namespaces / .inc fragments.
from __future__ import annotations

import re
from pathlib import Path

MAIN = Path(r"C:\Users\ethan\Codes\neuralnet.cpp")
ROOTS = [
    MAIN / "include",
    MAIN / "src",
    MAIN / "modules",
    MAIN / "engine" / "abstraction" / "include",
    MAIN / "engine" / "engine",
    MAIN / "engine" / "tests",
]
SKIP = {".git", "build", "exstl", ".playwright-mcp"}
TEXT = {".hpp", ".h", ".cpp", ".cc", ".inc"}

# order matters: longer / more specific first
REPLACEMENTS = [
    (re.compile(r"(?<![\w:])detail::fatal_at\b"), "exstl::detail::fatal_at"),
    (re.compile(r"(?<![\w:])detail::fail_abort\b"), "exstl::detail::fail_abort"),
    (re.compile(r"(?<![\w:])detail::fail_exit\b"), "exstl::detail::fail_exit"),
    (re.compile(r"(?<![\w:])detail::check_value\b"), "exstl::detail::check_value"),
    (re.compile(r"(?<![\w:])detail::check_exit\b"), "exstl::detail::check_exit"),
    (re.compile(r"(?<![\w:])detail::check_message\b"), "exstl::detail::check_message"),
    (re.compile(r"(?<![\w:])detail::contextualize\b"), "exstl::detail::contextualize"),
    (re.compile(r"(?<![\w:])parse_number\s*<"), "exstl::parse_number<"),
    (re.compile(r"(?<![\w:])parse_number\s*\("), "exstl::parse_number("),
    (re.compile(r"(?<![\w:])parallel_for_samples\s*\("), "exstl::parallel_for_samples("),
    (re.compile(r"(?<![\w:])parallel_for_blocks\s*\("), "exstl::parallel_for_blocks("),
    (re.compile(r"(?<![\w:])check_result\s*\("), "exstl::check_result("),
    # bare ThreadPool type usage: ThreadPool pool / ThreadPool&
    (re.compile(r"(?<![\w:])ThreadPool\s+(\w)"), r"exstl::ThreadPool \1"),
    (re.compile(r"(?<![\w:])ThreadPool&"), "exstl::ThreadPool&"),
    (re.compile(r"(?<![\w:])ThreadPool\*"), "exstl::ThreadPool*"),
]


def rewrite(text: str) -> str:
    for rx, rep in REPLACEMENTS:
        text = rx.sub(rep, text)
    text = text.replace("exstl::exstl::", "exstl::")
    text = text.replace("nn::exstl::", "exstl::")
    return text


def transform_blockwise(text: str, path: Path) -> str:
    # .inc: wholesale (always included inside nn)
    if path.suffix == ".inc":
        return rewrite(text)
    if "namespace nn" not in text:
        return rewrite(text) if ("parse_number" in text or "parallel_for" in text or "detail::fatal" in text) else text
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if re.search(r"\bnamespace\s+nn\b", line):
            depth = 0
            started = False
            j = i
            while j < n:
                cur = lines[j]
                opens = cur.count("{")
                closes = cur.count("}")
                if j == i and "{" in cur:
                    head, _, tail = cur.partition("{")
                    out.append(head + "{" + rewrite(tail))
                else:
                    out.append(rewrite(cur))
                depth += opens - closes
                started = started or opens > 0
                j += 1
                if started and depth <= 0:
                    break
            i = j
            continue
        out.append(line)
        i += 1
    return "".join(out)


def main():
    changed = 0
    for root in ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in TEXT:
                continue
            if any(part in SKIP for part in p.parts):
                continue
            try:
                raw = p.read_text(encoding="utf-8")
            except Exception:
                continue
            new = transform_blockwise(raw, p)
            if new != raw:
                p.write_text(new, encoding="utf-8", newline="\n")
                changed += 1
    print(f"changed {changed}")


if __name__ == "__main__":
    main()
