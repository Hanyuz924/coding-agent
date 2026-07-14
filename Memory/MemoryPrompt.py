"""Prompt templates for the background memory-extraction fork.

Python port of src/services/extractMemories/prompts.ts (auto-only variant).
The extraction fork runs as a perfect fork of the main conversation — same
system prompt, same message prefix — so this text is appended as a trailing
USER message (never the system prompt, which stays the parent's for cache
sharing). It fires only when the main agent didn't write memories itself.

Team-memory (combined) variant omitted intentionally; add TYPES_SECTION_COMBINED
+ the "never save API keys in shared memory" line if team memory is introduced.
"""

from __future__ import annotations

from Tools.BashTool.bash_prompt import (
    BASH_TOOL_NAME,
    FILE_EDIT_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_WRITE_TOOL_NAME,
    GLOB_TOOL_NAME,
    GREP_TOOL_NAME,
)

# ── frontmatter template (from memoryTypes.ts MEMORY_FRONTMATTER_EXAMPLE) ─────
MEMORY_TYPES = ["user", "feedback", "project", "reference"]

MEMORY_FRONTMATTER_EXAMPLE = f"""```markdown
---
name: {{{{memory name}}}}
description: {{{{one-line description — used to decide relevance in future conversations, so be specific}}}}
type: {{{{{", ".join(MEMORY_TYPES)}}}}}
---

{{{{memory content — for feedback/project types, structure as: rule/fact, then **Why:** and **How to apply:** lines}}}}
```"""

# ── the four-type taxonomy (from TYPES_SECTION_INDIVIDUAL) ────────────────────
TYPES_SECTION = """## Types of memory

There are several discrete types of memory that you can store in your memory system:

<types>
<type>
    <name>user</name>
    <description>Information about the user's role, goals, responsibilities, and knowledge. Great user memories let you tailor future behavior to the user's preferences and perspective — collaborate with a senior engineer differently than a first-time coder. Avoid memories that read as negative judgements or that aren't relevant to the work.</description>
    <when_to_save>When you learn any detail about the user's role, preferences, responsibilities, or knowledge.</when_to_save>
    <how_to_use>When your work should be informed by the user's profile or perspective — e.g. framing an explanation around what they already know.</how_to_use>
    <examples>
    user: I'm a data scientist investigating what logging we have in place
    assistant: [saves user memory: user is a data scientist, currently focused on observability/logging]
    </examples>
</type>
<type>
    <name>feedback</name>
    <description>Guidance the user has given about how to approach work — what to avoid AND what to keep doing. Record from failure and success: saving only corrections makes you drift from validated approaches and grow overly cautious.</description>
    <when_to_save>Any time the user corrects your approach ("no, not that", "stop doing X") OR confirms a non-obvious approach worked ("yes, exactly", accepting an unusual choice without pushback). Include *why* so you can judge edge cases later.</when_to_save>
    <how_to_use>Let these guide your behavior so the user doesn't have to give the same guidance twice.</how_to_use>
    <body_structure>Lead with the rule, then a **Why:** line (the reason given) and a **How to apply:** line (when it kicks in).</body_structure>
    <examples>
    user: don't mock the database in these tests — we got burned when mocked tests passed but the prod migration failed
    assistant: [saves feedback memory: integration tests must hit a real database, not mocks. Reason: prior mock/prod divergence masked a broken migration]
    </examples>
</type>
<type>
    <name>project</name>
    <description>Ongoing work, goals, initiatives, bugs, or incidents in the project that aren't derivable from the code or git history. Helps you understand the context and motivation behind the user's work.</description>
    <when_to_save>When you learn who is doing what, why, or by when. These states change fast — keep them current. Convert relative dates to absolute (e.g. "Thursday" → "2026-07-09").</when_to_save>
    <how_to_use>To understand the nuance behind a request and make better-informed suggestions.</how_to_use>
    <body_structure>Lead with the fact/decision, then a **Why:** line (motivation/constraint/deadline) and a **How to apply:** line (how it shapes your suggestions).</body_structure>
    <examples>
    user: we're freezing non-critical merges after Thursday — mobile is cutting a release branch
    assistant: [saves project memory: merge freeze begins 2026-07-09 for mobile release cut. Flag non-critical PR work scheduled after that date]
    </examples>
</type>
<type>
    <name>reference</name>
    <description>Pointers to where information lives in external systems, so you remember where to look for up-to-date info outside the project directory.</description>
    <when_to_save>When you learn about external resources and their purpose (e.g. bugs tracked in a specific Linear project, feedback in a specific Slack channel).</when_to_save>
    <how_to_use>When the user references an external system or info that may live in one.</how_to_use>
    <examples>
    user: check the Linear project "INGEST" for context on these tickets — that's where we track pipeline bugs
    assistant: [saves reference memory: pipeline bugs are tracked in Linear project "INGEST"]
    </examples>
</type>
</types>"""

# ── exclusions (from WHAT_NOT_TO_SAVE_SECTION) ────────────────────────────────
WHAT_NOT_TO_SAVE = """## What NOT to save in memory

- Code patterns, conventions, architecture, file paths, or project structure — derivable by reading the current project state.
- Git history, recent changes, or who-changed-what — `git log` / `git blame` are authoritative.
- Debugging solutions or fix recipes — the fix is in the code; the commit message has the context.
- Anything already documented in CLAUDE.md files.
- Ephemeral task details: in-progress work, temporary state, current conversation context.

These exclusions apply even when the user explicitly asks you to save. If they ask you to save a PR list or activity summary, ask what was *surprising* or *non-obvious* about it — that's the part worth keeping."""


def _opener(new_message_count: int, existing_memories: str, memory_dir: str) -> str:
    manifest = (
        f"\n\n## Existing memory files\n\n{existing_memories}\n\n"
        "Check this list before writing — update an existing file rather than "
        "creating a duplicate."
        if existing_memories
        else ""
    )
    return (
        f"You are now acting as the memory extraction subagent. Analyze the most "
        f"recent ~{new_message_count} messages above and use them to update your "
        f"persistent memory systems.\n\n"
        f"Memory directory: `{memory_dir}` — ALL files you write MUST be inside this "
        f"directory. Do not write to any other location.\n\n"
        f"Available tools: {FILE_READ_TOOL_NAME}, {GREP_TOOL_NAME}, {GLOB_TOOL_NAME}, "
        f"read-only {BASH_TOOL_NAME} (ls/find/cat/stat/wc/head/tail and similar), "
        f"and {FILE_EDIT_TOOL_NAME}/{FILE_WRITE_TOOL_NAME} for paths inside the memory "
        f"directory only. {BASH_TOOL_NAME} rm is not permitted. All other tools will be denied.\n\n"
        f"You have a limited turn budget. {FILE_EDIT_TOOL_NAME} requires a prior {FILE_READ_TOOL_NAME} of the same "
        f"file, so the efficient strategy is: turn 1 — issue all {FILE_READ_TOOL_NAME} calls in "
        f"parallel for every file you might update; turn 2 — issue all {FILE_WRITE_TOOL_NAME}/{FILE_EDIT_TOOL_NAME} "
        f"calls in parallel. Do not interleave reads and writes across turns.\n\n"
        f"You MUST only use content from the last ~{new_message_count} messages. "
        "Do not spend turns investigating or verifying that content further — no "
        "grepping source files, no reading code, no git commands." + manifest
    )


def _how_to_save(skip_index: bool) -> str:
    if skip_index:
        return (
            "## How to save memories\n\n"
            "Write each memory to its own file (e.g. `user_role.md`, "
            "`feedback_testing.md`) using this frontmatter format:\n\n"
            f"{MEMORY_FRONTMATTER_EXAMPLE}\n\n"
            "- Organize memory semantically by topic, not chronologically\n"
            "- Update or remove memories that turn out to be wrong or outdated\n"
            "- Do not write duplicates. First check for an existing memory to update."
        )
    return (
        "## How to save memories\n\n"
        "Saving a memory is a two-step process:\n\n"
        "**Step 1** — write the memory to its own file (e.g. `user_role.md`) using "
        f"this frontmatter format:\n\n{MEMORY_FRONTMATTER_EXAMPLE}\n\n"
        "**Step 2** — add a pointer in `MEMORY.md`. `MEMORY.md` is an index, not a "
        "memory — each entry is one line, under ~150 chars: "
        "`- [Title](file.md) — one-line hook`. It has no frontmatter. Never write "
        "memory content directly into `MEMORY.md`.\n\n"
        "- `MEMORY.md` is always loaded into your system prompt — keep it concise\n"
        "- Organize memory semantically by topic, not chronologically\n"
        "- Update or remove memories that turn out to be wrong or outdated\n"
        "- Do not write duplicates. First check for an existing memory to update."
    )


def build_extract_prompt(
    new_message_count: int,
    memory_dir: str,
    existing_memories: str = "",
    skip_index: bool = False,
) -> str:
    """Assemble the memory-extraction user-message prompt (auto-only variant).

    Pass the result as a trailing user message to the extraction fork:
        prompt_messages=[{"role": "user", "content": build_extract_prompt(...)}]
    """
    return "\n\n".join(
        [
            _opener(new_message_count, existing_memories, memory_dir),
            "If the user explicitly asks you to remember something, save it "
            "immediately as whichever type fits best. If they ask you to forget "
            "something, find and remove the relevant entry.",
            TYPES_SECTION,
            WHAT_NOT_TO_SAVE,
            _how_to_save(skip_index),
        ]
    )
