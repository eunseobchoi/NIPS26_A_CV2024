# Bounty Artifacts Index

This directory contains patch artifacts and audit reports produced as part of
bounty and research work on this repository.

---

## Artifacts

### `rustchain-1112/`
**File:** `ATTEST_SUBMIT_FUZZ_REPORT.md`  
Fuzz-testing attestation and submission report for the rustchain-1112 target.

---

### `archestra-3854/`
**File:** `REPORT.md`  
Audit report for the archestra-3854 bounty target.

---

### `claude-builders-3/`
**File:** `PATCH_ARTIFACT.md`  
Patch artifact for Claude Builders Bounty #3.  
Adds a Claude Code `PreToolUse` hook that blocks destructive Bash commands
(e.g. `rm -rf`, `dd of=/dev/sda`, `curl | bash`) before they execute.

**Issue:** https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/9  
**Bounty:** claude-builders-bounty/claude-builders-bounty#3  
**Files changed:**
- `.claude/hooks/pre_tool_use_bash_guard.py` — hook implementation
- `.claude/settings.json` — Claude Code hook wiring
- `tests/test_pre_tool_use_bash_guard.py` — unit tests (40+ cases)
