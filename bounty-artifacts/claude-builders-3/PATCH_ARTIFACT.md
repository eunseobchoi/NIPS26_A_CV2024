# Patch Artifact — Claude Builders #3
## Destructive Bash PreToolUse Hook

**Issue:** https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/9  
**Bounty reference:** claude-builders-bounty/claude-builders-bounty#3  
**Status:** Public patch artifact (not a completed `/opire try` submission — see note below)

---

## Problem Statement

Claude Code's autonomous Bash tool can execute shell commands without any
safety checkpoint.  A single hallucinated or malicious instruction containing
`rm -rf /`, `dd of=/dev/sda`, `curl … | bash`, or similar patterns could cause
**irreversible data loss** on the host machine.

The Claude Code `PreToolUse` hook mechanism allows a local script to intercept
every tool-use event before execution and emit a `block` or `approve` decision.
This patch exploits that mechanism to add a lightweight but comprehensive
destructive-command guard.

---

## Solution

### Files added

| Path | Purpose |
|------|---------|
| `.claude/hooks/pre_tool_use_bash_guard.py` | Hook script — pattern-matches the command and emits `block` or `approve` JSON |
| `.claude/settings.json` | Wires the hook into Claude Code for the `Bash` tool |
| `tests/test_pre_tool_use_bash_guard.py` | 40+ pytest unit tests covering blocked and approved commands |

### Hook protocol

