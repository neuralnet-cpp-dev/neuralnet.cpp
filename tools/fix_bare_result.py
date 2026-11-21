# Replace bare Result/Error inside namespace nn(...) with exstl:: qualified names.
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

RE_NN_NS = re.compile(r"\bnamespace\s+nn\b")
RE_RESULT = re.compile(r"(?<![\w:])Result\s*<")
RE_ERROR = re.compile(r"(?<![\w:])Error\b(?![_\w])")


def rewrite_segment(s: str) -> str:
    s = RE_RESULT.sub("exstl::Result<", s)
    s = RE_ERROR.sub("exstl::Error", s)
    return s


def transform(text: str, path: Path | None = None) -> str:
    # .inc fragments are included *inside* namespace nn by their parent header
    # and never declare the namespace themselves — rewrite them wholesale.
    if path is not None and path.suffix == ".inc":
        if RE_RESULT.search(text) or RE_ERROR.search(text):
            return rewrite_segment(text)
    if not RE_NN_NS.search(text):
        return text
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if RE_NN_NS.search(line):
            depth = 0
            started = False
            j = i
            while j < n:
                cur = lines[j]
                opens = cur.count("{")
                closes = cur.count("}")
                if j == i:
                    if "{" in cur:
                        head, _, tail = cur.partition("{")
                        out.append(head + "{" + rewrite_segment(tail))
                    else:
                        out.append(cur)
                else:
                    out.append(rewrite_segment(cur))
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


def iter_files():
    for root in ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in TEXT:
                continue
            if any(part in SKIP for part in p.parts):
                continue
            yield p


def main():
    changed = 0
    scanned = 0
    for p in iter_files():
        try:
            raw = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if p.suffix == ".inc" or RE_NN_NS.search(raw):
            scanned += 1
            new = transform(raw, p)
            if new != raw:
                p.write_text(new, encoding="utf-8", newline="\n")
                changed += 1
    print(f"scanned {scanned}, changed {changed}")


if __name__ == "__main__":
    main()
