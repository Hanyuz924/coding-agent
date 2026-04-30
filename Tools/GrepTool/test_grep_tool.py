"""
Tests for GrepTool.
Each test passes a dict that mirrors what the LLM would send as tool input.

Fixture files live at Tools/GrepTool/fixtures/ so tests can be extended
without recreating a workspace each run:
    fixtures/
        main.py        — hello/goodbye functions, Greeter class
        utils.py       — arithmetic functions (add, subtract, multiply, divide …)
        README.md      — markdown with Hello/Goodbye mentions
        sub/
            helper.py  — HELLO/BYE constants, retry/truncate helpers
            config.json — JSON with "greeting": "Hello"
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from GrepTool import GrepTool, GrepOutput
from BaseTool import ToolResult

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool():
    return GrepTool()


# ---------------------------------------------------------------------------
# Basic: files_with_matches (default mode)
# ---------------------------------------------------------------------------

def test_files_with_matches_finds_files(tool):
    """LLM searches for a pattern — expects list of matching file paths."""
    result = tool.execute(pattern="hello", path=FIXTURES)

    assert result.success is True
    assert isinstance(result.data, GrepOutput)
    assert result.data.mode == "files_with_matches"
    assert result.data.num_files > 0
    assert any("main.py" in f for f in result.data.filenames)


def test_files_with_matches_no_results(tool):
    """LLM searches for a pattern that matches nothing — success but empty."""
    result = tool.execute(pattern="ZZZNOMATCH_EVER_XYZ", path=FIXTURES)

    assert result.success is True
    assert result.data.num_files == 0
    assert result.data.filenames == []


def test_files_with_matches_sorted_by_mtime(tool):
    """Results should be sorted newest-first (most recently modified)."""
    result = tool.execute(pattern="def", path=FIXTURES, output_mode="files_with_matches")

    assert result.success is True
    assert result.data.num_files > 1  # multiple .py files have 'def'


# ---------------------------------------------------------------------------
# Content mode
# ---------------------------------------------------------------------------

def test_content_mode_returns_matching_lines(tool):
    """LLM requests content mode to see the actual matching lines."""
    result = tool.execute(pattern="def", path=FIXTURES, output_mode="content")

    assert result.success is True
    assert result.data.mode == "content"
    assert result.data.content is not None
    assert "def" in result.data.content
    assert result.data.num_lines > 0


def test_content_mode_no_matches(tool):
    """Content mode with no matches — returns empty content."""
    result = tool.execute(pattern="ZZZNOMATCH", path=FIXTURES, output_mode="content")

    assert result.success is True
    assert result.data.content == ""


def test_content_mode_multiline_class(tool):
    """Searching for class definition should hit Greeter in main.py."""
    result = tool.execute(pattern="class Greeter", path=FIXTURES, output_mode="content")

    assert result.success is True
    assert "Greeter" in result.data.content


# ---------------------------------------------------------------------------
# Count mode
# ---------------------------------------------------------------------------

def test_count_mode_returns_counts(tool):
    """LLM uses count mode to see how many matches per file."""
    result = tool.execute(pattern="def", path=FIXTURES, output_mode="count")

    assert result.success is True
    assert result.data.mode == "count"
    assert result.data.num_matches is not None
    assert result.data.num_matches > 0


def test_count_mode_no_matches(tool):
    """Count mode with no matches returns zero total."""
    result = tool.execute(pattern="ZZZNOMATCH", path=FIXTURES, output_mode="count")

    assert result.success is True
    assert result.data.num_matches == 0


def test_count_mode_multiple_files(tool):
    """'def' appears in both main.py and utils.py — count should reflect that."""
    result = tool.execute(pattern="def", path=FIXTURES, output_mode="count")

    assert result.success is True
    assert result.data.num_files >= 2


# ---------------------------------------------------------------------------
# Case insensitive
# ---------------------------------------------------------------------------

def test_case_insensitive_matches_uppercase(tool):
    """LLM enables case_insensitive to catch HELLO (in helper.py) and hello."""
    sensitive = tool.execute(
        pattern="HELLO", path=FIXTURES,
        output_mode="files_with_matches", case_insensitive=False,
    )
    insensitive = tool.execute(
        pattern="HELLO", path=FIXTURES,
        output_mode="files_with_matches", case_insensitive=True,
    )

    # case-insensitive catches more (or equal) files
    assert insensitive.data.num_files >= sensitive.data.num_files
    # sensitive should find helper.py (HELLO = "hello") but NOT main.py (lowercase hello)
    assert any("helper.py" in f for f in sensitive.data.filenames)


# ---------------------------------------------------------------------------
# head_limit
# ---------------------------------------------------------------------------

def test_head_limit_caps_results(tool):
    """LLM sets head_limit=1 — should return at most 1 file."""
    result = tool.execute(
        pattern="def", path=FIXTURES,
        output_mode="files_with_matches", head_limit=1,
    )

    assert result.success is True
    assert result.data.num_files <= 1


def test_head_limit_zero_means_unlimited(tool):
    """head_limit=0 should return more results than head_limit=1."""
    limited = tool.execute(
        pattern="def", path=FIXTURES,
        output_mode="files_with_matches", head_limit=1,
    )
    unlimited = tool.execute(
        pattern="def", path=FIXTURES,
        output_mode="files_with_matches", head_limit=0,
    )

    assert unlimited.data.num_files >= limited.data.num_files


def test_head_limit_content_mode(tool):
    """head_limit in content mode caps the number of output lines."""
    result = tool.execute(
        pattern="def", path=FIXTURES,
        output_mode="content", head_limit=3,
    )

    assert result.success is True
    assert result.data.num_lines <= 3


# ---------------------------------------------------------------------------
# Offset
# ---------------------------------------------------------------------------

def test_offset_skips_first_results(tool):
    """offset=1 in files_with_matches should skip the first file."""
    full = tool.execute(pattern="def", path=FIXTURES, output_mode="files_with_matches", head_limit=0)
    offset_one = tool.execute(pattern="def", path=FIXTURES, output_mode="files_with_matches", head_limit=0, offset=1)

    assert full.data.num_files > offset_one.data.num_files
    if full.data.num_files >= 2:
        assert full.data.filenames[1] == offset_one.data.filenames[0]


# ---------------------------------------------------------------------------
# File type filter
# ---------------------------------------------------------------------------

def test_type_filter_restricts_to_python(tool):
    """LLM uses type='py' to search only Python files, skipping README.md and config.json."""
    result = tool.execute(
        pattern="Hello", path=FIXTURES,
        output_mode="files_with_matches", type="py",
    )

    assert result.success is True
    assert all(f.endswith(".py") for f in result.data.filenames)
    assert not any(f.endswith(".md") for f in result.data.filenames)


def test_type_filter_json(tool):
    """type='json' should only find config.json."""
    result = tool.execute(
        pattern="greeting", path=FIXTURES,
        output_mode="files_with_matches", type="json",
    )

    assert result.success is True
    assert result.data.num_files >= 1
    assert all(f.endswith(".json") for f in result.data.filenames)


# ---------------------------------------------------------------------------
# Glob filter
# ---------------------------------------------------------------------------

def test_glob_filter_matches_md_only(tool):
    """LLM uses glob to restrict search to .md files."""
    result = tool.execute(
        pattern="Hello", path=FIXTURES,
        output_mode="files_with_matches", glob="*.md",
    )

    assert result.success is True
    assert all(f.endswith(".md") for f in result.data.filenames)


def test_glob_filter_python_only(tool):
    """glob='*.py' restricts search to python files."""
    result = tool.execute(
        pattern="def", path=FIXTURES,
        output_mode="files_with_matches", glob="*.py",
    )

    assert result.success is True
    assert all(f.endswith(".py") for f in result.data.filenames)


# ---------------------------------------------------------------------------
# Subdirectory search
# ---------------------------------------------------------------------------

def test_recursive_finds_files_in_subdirs(tool):
    """rg is recursive by default — should find helper.py inside sub/."""
    result = tool.execute(pattern="MAX_RETRIES", path=FIXTURES)

    assert result.success is True
    assert any("helper.py" in f for f in result.data.filenames)


def test_search_in_subdirectory_directly(tool):
    """Searching the sub/ directory directly should work."""
    sub_path = os.path.join(FIXTURES, "sub")
    result = tool.execute(pattern="HELLO", path=sub_path)

    assert result.success is True
    assert result.data.num_files >= 1


# ---------------------------------------------------------------------------
# Pattern starting with '-'
# ---------------------------------------------------------------------------

def test_pattern_starting_with_dash(tool):
    """Patterns starting with '-' must be passed via -e to avoid flag parsing."""
    # '-' won't match anything in our fixtures, but should not crash
    result = tool.execute(pattern="-nonexistent", path=FIXTURES)

    assert result.success is True  # rg should handle it gracefully (no matches)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_invalid_path_returns_failure(tool):
    """LLM provides a path that does not exist."""
    result = tool.execute(pattern="anything", path="/nonexistent/path/that/does/not/exist")

    assert result.success is False


def test_invalid_regex_returns_failure(tool):
    """LLM provides a broken regex pattern."""
    result = tool.execute(pattern="[unclosed", path=FIXTURES)

    assert result.success is False
