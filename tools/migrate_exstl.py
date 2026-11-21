# One-shot migration: foundation symbols/includes -> exstl
# Run from anywhere; absolute paths below.
from __future__ import annotations

import re
from pathlib import Path

MAIN = Path(r"C:\Users\ethan\Codes\neuralnet.cpp")
ENGINE = MAIN / "engine"
EXSTL = MAIN / "exstl"

SKIP_DIRS = {".git", "build", "exstl", ".playwright-mcp", "CMakeFiles", ".cache"}
SKIP_SUFFIX = {".png", ".jpg", ".pdf", ".bin", ".spv", ".nnk", ".nndataset", ".nnvocab", ".zip"}

# ── include path map (longest / most specific first) ─────────────────────────
INCLUDE_MAP = [
    ('#include <exstl/errors.hpp>', '#include <exstl/errors.hpp>'),
    ('#include <exstl/errors.hpp>', '#include <exstl/errors.hpp>'),
    ('#include <exstl/assert.hpp>', '#include <exstl/assert.hpp>'),
    ('#include <exstl/assert.hpp>', '#include <exstl/assert.hpp>'),
    ('#include <exstl/threadpool.hpp>', '#include <exstl/threadpool.hpp>'),
    ('#include <exstl/threadpool.hpp>', '#include <exstl/threadpool.hpp>'),
    ('#include <exstl/parallel.hpp>', '#include <exstl/parallel.hpp>'),
    ('#include <exstl/parallel.hpp>', '#include <exstl/parallel.hpp>'),
    ('#include <exstl/observer_ptr.hpp>', '#include <exstl/observer_ptr.hpp>'),
    ('#include <exstl/observer_ptr.hpp>', '#include <exstl/observer_ptr.hpp>'),
    ('#include <exstl/error_macros.hpp>', '#include <exstl/error_macros.hpp>'),
    ('#include <exstl/error_macros.hpp>', '#include <exstl/error_macros.hpp>'),
    ('#include <exstl/error_macros.hpp>', '#include <exstl/error_macros.hpp>'),
    ('#include <exstl/errors.hpp>', '#include <exstl/errors.hpp>'),
    ('#include <exstl/assert.hpp>', '#include <exstl/assert.hpp>'),
    ('#include <exstl/threadpool.hpp>', '#include <exstl/threadpool.hpp>'),
    ('#include <exstl/parallel.hpp>', '#include <exstl/parallel.hpp>'),
    ('#include <exstl/observer_ptr.hpp>', '#include <exstl/observer_ptr.hpp>'),
]

# ── macro renames (foundation only; longest first) ───────────────────────────
MACRO_MAP = [
    ("EX_EASY_CHECK", "EX_EASY_CHECK"),
    ("EX_ASSERT_ALWAYS", "EX_ASSERT_ALWAYS"),
    ("EX_TRY_CHECK", "EX_TRY_CHECK"),
    ("EX_TRY_MSG", "EX_TRY_MSG"),
    ("EX_UNWRAP_MSG", "EX_UNWRAP_MSG"),
    ("EX_UNWRAP_1", "EX_UNWRAP_1"),
    ("EX_DETAIL_PICK2", "EX_DETAIL_PICK2"),
    ("EX_DETAIL_PICK3", "EX_DETAIL_PICK3"),
    ("EX_EXIT_MSG", "EX_EXIT_MSG"),
    ("EX_EXIT_2", "EX_EXIT_2"),
    ("EX_HOT_FN", "EX_HOT_FN"),
    ("EX_LIKELY", "EX_LIKELY"),
    ("EX_UNLIKELY", "EX_UNLIKELY"),
    ("EX_CPU_PAUSE", "EX_CPU_PAUSE"),
    ("EX_ASSERT", "EX_ASSERT"),
    ("EX_REQUIRE", "EX_REQUIRE"),
    ("EX_FATAL", "EX_FATAL"),
    ("EX_UNWRAP", "EX_UNWRAP"),
    ("EX_CHECK", "EX_CHECK"),
    ("EX_EXIT", "EX_EXIT"),
    ("EX_TRY", "EX_TRY"),
    ("EX_FAIL", "EX_FAIL"),
    ("EX_PICK_END", "EX_PICK_END"),
]

# ── qualified symbol renames (foundation only; longest first) ────────────────
# env raw layer: nn::detail::env_* -> exstl::env::{flag,str,integer}
SYMBOL_MAP = [
    ("exstl::detail::check_value", "exstl::detail::check_value"),
    ("exstl::detail::check_exit", "exstl::detail::check_exit"),
    ("exstl::detail::check_message", "exstl::detail::check_message"),
    ("exstl::detail::contextualize", "exstl::detail::contextualize"),
    ("exstl::detail::fatal_at", "exstl::detail::fatal_at"),
    ("exstl::detail::fail_abort", "exstl::detail::fail_abort"),
    ("exstl::detail::fail_exit", "exstl::detail::fail_exit"),
    ("exstl::detail::PoolTask", "exstl::detail::PoolTask"),
    ("exstl::detail::RegionState", "exstl::detail::RegionState"),
    ("exstl::detail::invoke_nothrow", "exstl::detail::invoke_nothrow"),
    ("exstl::env::flag", "exstl::env::flag"),
    ("exstl::env::str", "exstl::env::str"),
    ("exstl::env::integer", "exstl::env::integer"),
    ("exstl::global_thread_pool", "exstl::global_thread_pool"),
    ("exstl::PARALLEL_THRESHOLD", "exstl::PARALLEL_THRESHOLD"),
    ("exstl::parallel_for_blocks", "exstl::parallel_for_blocks"),
    ("exstl::parallel_for_samples", "exstl::parallel_for_samples"),
    ("exstl::transform_reduce", "exstl::transform_reduce"),
    ("exstl::make_observer", "exstl::make_observer"),
    ("exstl::observer_ptr", "exstl::observer_ptr"),
    ("exstl::check_result", "exstl::check_result"),
    ("exstl::parse_number", "exstl::parse_number"),
    ("exstl::ThreadPool", "exstl::ThreadPool"),
    ("exstl::for_each", "exstl::for_each"),
    # exstl::transform — only the parallel algorithm (followed by ( or space)
    # handled via regex below
    ("exstl::Error", "exstl::Error"),
    ("exstl::Result", "exstl::Result"),
]

TEXT_EXTS = {".hpp", ".h", ".cpp", ".cc", ".md", ".ps1", ".py", ".cmake", ".txt", ".json"}


def transform(text: str) -> str:
    for old, new in INCLUDE_MAP:
        text = text.replace(old, new)
    for old, new in MACRO_MAP:
        # word-boundary-ish: avoid EX_TRY matching inside longer protected names
        # (our map is longest-first among foundation macros; protected macros
        #  like NN_ENABLE_PROBES never appear as exact MACRO_MAP keys)
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    for old, new in SYMBOL_MAP:
        text = text.replace(old, new)
    # exstl::transform( or exstl::transform< — not exstl::transform_reduce (already done)
    text = re.sub(r"\bnn::transform\b(?!\s*_)", "exstl::transform", text)
    # leftover bare macros in comments already handled; fix "EX_ASSERT failed" strings
    text = text.replace('"EX_ASSERT failed', '"EX_ASSERT failed')
    text = text.replace('"EX_ASSERT_ALWAYS"', '"EX_ASSERT_ALWAYS"')
    text = text.replace('"EX_FATAL"', '"EX_FATAL"')
    text = text.replace("[exstl]", "[exstl]")  # error print prefix in comments/strings
    return text


def iter_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        # don't rewrite exstl's own copies
        if EXSTL in p.parents or p == EXSTL:
            continue
        yield p


def migrate_tree(root: Path, label: str) -> tuple[int, int]:
    files = 0
    changed = 0
    for p in iter_files(root):
        try:
            raw = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        files += 1
        new = transform(raw)
        if new != raw:
            p.write_text(new, encoding="utf-8", newline="\n")
            changed += 1
    print(f"[{label}] scanned {files}, changed {changed}")
    return files, changed


def main():
    # Only migrate code trees (not docs/history for now — doc_audit later)
    for sub in ("include", "src", "modules", "tools", "bench", "examples"):
        d = MAIN / sub
        if d.is_dir():
            migrate_tree(d, f"main/{sub}")
    # engine: abstraction + engine impl + tests + CMake fragments as text
    for sub in ("abstraction", "engine", "tests", "docs"):
        d = ENGINE / sub
        if d.is_dir():
            # skip engine/docs to avoid rewriting design history mid-flight
            if sub == "docs":
                continue
            migrate_tree(d, f"engine/{sub}")
    # engine root CMakeLists
    cmake = ENGINE / "CMakeLists.txt"
    if cmake.is_file():
        t = cmake.read_text(encoding="utf-8")
        n = transform(t)
        if n != t:
            cmake.write_text(n, encoding="utf-8", newline="\n")
            print("[engine] CMakeLists.txt changed")
    # main CMakeLists — only symbol/include parts; we'll edit structure manually
    # (transform still useful for file-list paths)
    main_cmake = MAIN / "CMakeLists.txt"
    if main_cmake.is_file():
        t = main_cmake.read_text(encoding="utf-8")
        n = transform(t)
        # also rewrite file-list paths for deleted headers
        n = n.replace(
            "engine/abstraction/include/neuralnet.cpp/core_assert.hpp",
            "exstl/include/exstl/assert.hpp",
        )
        n = n.replace(
            "engine/abstraction/include/neuralnet.cpp/core_errors.hpp",
            "exstl/include/exstl/errors.hpp",
        )
        n = n.replace(
            "engine/abstraction/include/neuralnet.cpp/core_threadpool.hpp",
            "exstl/include/exstl/threadpool.hpp",
        )
        n = n.replace(
            "engine/abstraction/include/neuralnet.cpp/core_parallel.hpp",
            "exstl/include/exstl/parallel.hpp",
        )
        n = n.replace(
            "engine/abstraction/include/neuralnet.cpp/core_observer_ptr.hpp",
            "exstl/include/exstl/observer_ptr.hpp",
        )
        n = n.replace(
            "engine/abstraction/include/nn_error_macros.hpp",
            "exstl/include/exstl/error_macros.hpp",
        )
        if n != t:
            main_cmake.write_text(n, encoding="utf-8", newline="\n")
            print("[main] CMakeLists.txt changed")


if __name__ == "__main__":
    main()
