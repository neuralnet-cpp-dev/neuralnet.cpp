from pathlib import Path
import re

ROOTS = [
    Path(r"C:\Users\ethan\Codes\neuralnet.cpp\engine"),
    Path(r"C:\Users\ethan\Codes\neuralnet.cpp\include"),
    Path(r"C:\Users\ethan\Codes\neuralnet.cpp\src"),
    Path(r"C:\Users\ethan\Codes\neuralnet.cpp\modules"),
]
SKIP = {".git", "build", "exstl", ".playwright-mcp"}

MACRO_MAP = [
    ("NN_EASY_CHECK", "EX_EASY_CHECK"),
    ("NN_ASSERT_ALWAYS", "EX_ASSERT_ALWAYS"),
    ("NN_TRY_CHECK", "EX_TRY_CHECK"),
    ("NN_TRY_MSG", "EX_TRY_MSG"),
    ("NN_UNWRAP_MSG", "EX_UNWRAP_MSG"),
    ("NN_UNWRAP_1", "EX_UNWRAP_1"),
    ("NN_DETAIL_PICK2", "EX_DETAIL_PICK2"),
    ("NN_DETAIL_PICK3", "EX_DETAIL_PICK3"),
    ("NN_EXIT_MSG", "EX_EXIT_MSG"),
    ("NN_EXIT_2", "EX_EXIT_2"),
    ("NN_HOT_FN", "EX_HOT_FN"),
    ("NN_LIKELY", "EX_LIKELY"),
    ("NN_UNLIKELY", "EX_UNLIKELY"),
    ("NN_CPU_PAUSE", "EX_CPU_PAUSE"),
    ("NN_ASSERT", "EX_ASSERT"),
    ("NN_REQUIRE", "EX_REQUIRE"),
    ("NN_FATAL", "EX_FATAL"),
    ("NN_UNWRAP", "EX_UNWRAP"),
    ("NN_CHECK", "EX_CHECK"),
    ("NN_EXIT", "EX_EXIT"),
    ("NN_TRY", "EX_TRY"),
    ("NN_FAIL", "EX_FAIL"),
    ("NN_PICK_END", "EX_PICK_END"),
]
SYMBOL_MAP = [
    ("nn::observer_ptr", "exstl::observer_ptr"),
    ("nn::make_observer", "exstl::make_observer"),
    ("nn::ThreadPool", "exstl::ThreadPool"),
    ("nn::global_thread_pool", "exstl::global_thread_pool"),
    ("nn::PARALLEL_THRESHOLD", "exstl::PARALLEL_THRESHOLD"),
    ("nn::parse_number", "exstl::parse_number"),
]

changed = 0
for root in ROOTS:
    if not root.is_dir():
        continue
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".hpp", ".h", ".cpp", ".cc", ".inc"}:
            continue
        if any(part in SKIP for part in p.parts):
            continue
        try:
            raw = p.read_text(encoding="utf-8")
        except Exception:
            continue
        text = raw
        for old, new in MACRO_MAP:
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        for old, new in SYMBOL_MAP:
            text = text.replace(old, new)
        text = re.sub(r"(?<![\w:])observer_ptr\s*<", "exstl::observer_ptr<", text)
        text = re.sub(r"(?<![\w:])make_observer\s*\(", "exstl::make_observer(", text)
        text = re.sub(r"(?<![\w:])global_thread_pool\s*\(", "exstl::global_thread_pool(", text)
        text = re.sub(r"(?<![\w:])PARALLEL_THRESHOLD\b", "exstl::PARALLEL_THRESHOLD", text)
        text = text.replace("exstl::exstl::", "exstl::")
        if text != raw:
            p.write_text(text, encoding="utf-8", newline="\n")
            changed += 1
print(f"changed {changed}")
