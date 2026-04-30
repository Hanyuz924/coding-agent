"""
Tests for BashTool.
Each test passes a dict that mirrors what the LLM would send as tool input.

Avoids: find, ls, grep, cat, head, tail, sed, awk, echo
Uses: pwd, wc, sort, mkdir, touch, python3, git, true/false, env, etc.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from BashTool import BashTool
from BaseTool import ToolResult


@pytest.fixture
def tool():
    return BashTool()


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

def test_exit_zero_is_success(tool):
    result = tool.execute(command="true")

    assert result.success is True
    assert result.metadata["returncode"] == 0


def test_exit_nonzero_is_failure(tool):
    result = tool.execute(command="false")

    assert result.success is False
    assert result.metadata["returncode"] != 0


def test_returns_tool_result(tool):
    result = tool.execute(command="true")

    assert isinstance(result, ToolResult)


# ---------------------------------------------------------------------------
# stdout / stderr capture
# ---------------------------------------------------------------------------

def test_stdout_captured(tool):
    result = tool.execute(command="python3 -c \"print('stdout_marker')\"")

    assert result.success is True
    assert "stdout_marker" in result.data


def test_stderr_captured(tool):
    result = tool.execute(command="python3 -c \"import sys; sys.stderr.write('stderr_marker\\n')\"")

    assert "stderr_marker" in result.data


def test_stdout_and_stderr_combined(tool):
    result = tool.execute(
        command="python3 -c \"import sys; print('out'); sys.stderr.write('err\\n')\""
    )

    assert "out" in result.data
    assert "err" in result.data


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------

def test_ls_lists_files_in_directory(tool, tmp_path):
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "beta.txt").write_text("b")

    result = tool.execute(command=f"ls {tmp_path}")

    assert result.success is True
    assert "alpha.txt" in result.data
    assert "beta.txt" in result.data


def test_ls_long_format(tool, tmp_path):
    (tmp_path / "file.txt").write_text("content")

    result = tool.execute(command=f"ls -l {tmp_path}")

    assert result.success is True
    assert "file.txt" in result.data


def test_ls_subdirectory(tool, tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "inner.txt").write_text("x")

    result = tool.execute(command=f"ls {sub}")

    assert result.success is True
    assert "inner.txt" in result.data


def test_ls_nonexistent_path_fails(tool):
    result = tool.execute(command="ls /nonexistent/path/xyz")

    assert result.success is False


def test_ls_shows_hidden_files(tool, tmp_path):
    (tmp_path / ".hidden").write_text("h")
    (tmp_path / "visible.txt").write_text("v")

    result = tool.execute(command=f"ls -a {tmp_path}")

    assert result.success is True
    assert ".hidden" in result.data
    assert "visible.txt" in result.data


# ---------------------------------------------------------------------------
# Common shell commands
# ---------------------------------------------------------------------------

def test_pwd_returns_absolute_path(tool):
    result = tool.execute(command="pwd")

    assert result.success is True
    assert result.data.startswith("/")


def test_python_arithmetic(tool):
    result = tool.execute(command="python3 -c \"print(3 + 4)\"")

    assert result.success is True
    assert "7" in result.data


def test_wc_counts_lines(tool, tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("line1\nline2\nline3\n")

    result = tool.execute(command=f"wc -l < {target}")

    assert result.success is True
    assert "3" in result.data


def test_sort_orders_output(tool, tmp_path):
    target = tmp_path / "unsorted.txt"
    target.write_text("banana\napple\ncherry\n")

    result = tool.execute(command=f"sort {target}")

    assert result.success is True
    lines = [l for l in result.data.splitlines() if l]
    assert lines == sorted(lines)


def test_env_variable_expansion(tool):
    result = tool.execute(command="export MYVAR=hello_world && python3 -c \"import os; print(os.environ['MYVAR'])\"")

    assert result.success is True
    assert "hello_world" in result.data


def test_pipe_with_wc(tool):
    result = tool.execute(command="python3 -c \"print('a\\nb\\nc')\" | wc -l")

    assert result.success is True
    assert "3" in result.data


def test_multiline_via_semicolon(tool):
    result = tool.execute(
        command="python3 -c \"print('first')\"; python3 -c \"print('second')\""
    )

    assert result.success is True
    assert "first" in result.data
    assert "second" in result.data


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------

def test_absolute_path_read(tool, tmp_path):
    target = tmp_path / "hello.txt"
    target.write_text("fixture content\n")

    result = tool.execute(command=f"python3 -c \"print(open('{target}').read(), end='')\"")

    assert result.success is True
    assert "fixture content" in result.data


def test_path_with_spaces(tool, tmp_path):
    spaced = tmp_path / "my dir"
    spaced.mkdir()
    note = spaced / "note.txt"
    note.write_text("spaced path works\n")

    result = tool.execute(command=f"python3 -c \"print(open('{note}').read(), end='')\"")

    assert result.success is True
    assert "spaced path works" in result.data


def test_write_then_read_back(tool, tmp_path):
    target = tmp_path / "out.txt"

    write_result = tool.execute(
        command=f"python3 -c \"open('{target}', 'w').write('written_content')\""
    )
    read_result = tool.execute(
        command=f"python3 -c \"print(open('{target}').read())\""
    )

    assert write_result.success is True
    assert read_result.success is True
    assert "written_content" in read_result.data


def test_mkdir_creates_directory(tool, tmp_path):
    new_dir = tmp_path / "newdir"

    tool.execute(command=f"mkdir {new_dir}")

    assert new_dir.is_dir()


def test_touch_creates_file(tool, tmp_path):
    new_file = tmp_path / "touched.txt"

    tool.execute(command=f"touch {new_file}")

    assert new_file.exists()


# ---------------------------------------------------------------------------
# Chained commands with &&
# ---------------------------------------------------------------------------

def test_chain_stops_on_failure(tool):
    result = tool.execute(command="false && python3 -c \"print('should_not_print')\"")

    assert result.success is False
    assert "should_not_print" not in result.data


def test_chain_continues_on_success(tool):
    result = tool.execute(command="true && python3 -c \"print('after_true')\"")

    assert result.success is True
    assert "after_true" in result.data


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_timeout_kills_long_command(tool):
    result = tool.execute(command="sleep 10", timeout=200)

    assert result.success is False
    assert result.metadata.get("timed_out") is True


def test_fast_command_completes_within_timeout(tool):
    result = tool.execute(command="python3 -c \"print('fast')\"", timeout=5000)

    assert result.success is True
    assert "fast" in result.data


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_nonexistent_command_fails(tool):
    result = tool.execute(command="this_cmd_does_not_exist_xyz")

    assert result.success is False


def test_missing_file_fails(tool):
    result = tool.execute(command="python3 /nonexistent/path/file.py")

    assert result.success is False


def test_description_param_ignored_in_output(tool):
    """description is metadata for the LLM — should not affect execution."""
    result = tool.execute(command="python3 -c \"print('ok')\"", description="Run python snippet")

    assert result.success is True
    assert "ok" in result.data


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

def test_large_output_is_truncated(tool):
    """20 000-char limit: generate ~1 MB of output and verify truncation flag."""
    result = tool.execute(command="python3 -c \"print('x' * 1_100_000)\"")

    assert result.metadata.get("truncated") is True
    assert len(result.data) <= 20_200


# ---------------------------------------------------------------------------
# Run pytest itself via BashTool
# ---------------------------------------------------------------------------

def test_run_pytest_via_bash(tool):
    """BashTool can invoke pytest on GrepTool's test suite and capture output."""
    result = tool.execute(
        command="cd /home/chenguanxi/coding-agent/Tools/GrepTool && python3 -m pytest test_grep_tool.py -v -s",
        description="Run GrepTool test suite via BashTool",
    )
    print(result.data)
    assert result.success is True
    assert "passed" in result.data
