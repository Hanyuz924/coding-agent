---
name: Verify claims before affirming
description: Don't say "You're right" at conversation start without first verifying the claim is actually correct
type: feedback
---

Don't open responses with affirmations like "You're right" or "Absolutely" unless you've first researched/verified the claim. More broadly: in code reviews and technical analysis, verify claims thoroughly before stating them as fact.

**Why:** Premature validation without checking facts undermines credibility and wastes the user's time. In code reviews, incomplete analysis leads to false "issues" that damage the quality of feedback. Specific example: I marked "context_tokens implementation uses heuristic estimation" as a MAJOR ISSUE without tracing actual API token flow (agent.py:349-354 uses real tokens, not estimates). The user's single-word prompt "context_tokens" signaled I'd gotten it wrong.

**How to apply:** When responding to a claim or correction: (1) investigate/verify first by reading actual code flow, not just snippets, (2) then respond with confirmation if correct or correction if not, (3) base acknowledgment on actual evidence. In code reviews: read the full execution path before marking something broken. This applies especially to technical claims about code behavior, architecture, or implementation correctness.
