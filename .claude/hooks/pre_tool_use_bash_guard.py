#!/usr/bin/env python3
"""
Claude Code PreToolUse hook — Destructive Bash Guard
=====================================================
Intercepts every Bash tool-use event before execution.
If the command matches a known-destructive pattern the hook
emits a JSON ``block`` response and exits non-zero so Claude
Code surfaces the refusal to the user.

Wire-up (already present in .claude/settings.json):

    {
      "hooks": {
        "PreToolUse": [
          {
            "matcher": "Bash",
            "hooks": [
              {
                "type": "command",
                "command": "python3 .claude/hooks/pre_tool_use_bash_guard.py"
              }
            ]
          }
        ]
      }
    }

Protocol
--------
Input  (stdin) : JSON object – the full tool-use payload supplied by
                 Claude Code.  The ``input`` field contains the
                 arguments forwarded to the tool, e.g.
                 ``{"command": "rm -rf /tmp/foo"}``.

Output (stdout): JSON object consumed by Claude Code:
  - ``{"decision": "approve"}``                  — allow the command
  - ``{"decision": "block", "reason": "<text>"}``— refuse the command

Exit code 0 is always used so that the hook runner itself does not
produce a noisy error; the ``block`` decision is the semantic signal.
"""

from __future__ import annotations

import json
import re
import sys
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Destructive-pattern registry
# ---------------------------------------------------------------------------

class _Pattern(NamedTuple):
    regex: re.Pattern[str]
    label: str
    explanation: str


# Each entry is (compiled-regex, short-label, human-readable-explanation).
# Patterns are evaluated in order; the first match wins.
_DESTRUCTIVE_PATTERNS: list[_Pattern] = [
    _Pattern(
        re.compile(r"\brm\s+.*-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\b|\brm\s+.*-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\b|\brm\s+-rf\b|\brm\s+-fr\b"),
        "rm -rf",
        "`rm -rf` (or equivalent flag ordering) can permanently delete entire "
        "directory trees with no recovery path.",
    ),
    _Pattern(
        re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*--no-preserve-root"),
        "rm --no-preserve-root",
        "`rm --no-preserve-root` bypasses the last safeguard protecting `/`.",
    ),
    _Pattern(
        # Matches: rm -r /  or  rm -r / (with trailing slash variants)
        re.compile(r"\brm\b[^|;&\n]*-[a-zA-Z]*r[a-zA-Z]*\s+[\"']?/[\"']?\s*($|[|;&])"),
        "rm -r /",
        "`rm -r /` would attempt to delete the entire filesystem root.",
    ),
    _Pattern(
        re.compile(r"\bdd\b.*\bof=/dev/[a-zA-Z]"),
        "dd to block device",
        "`dd of=/dev/<device>` writes raw data directly to a block device and "
        "can corrupt or erase disks.",
    ),
    _Pattern(
        re.compile(r"\bmkfs\b"),
        "mkfs",
        "`mkfs` formats a filesystem, destroying all existing data on the target.",
    ),
    _Pattern(
        re.compile(r"\bfdisk\b|\bparted\b|\bgdisk\b"),
        "partition editor",
        "Partition editors (`fdisk`, `parted`, `gdisk`) can rewrite partition "
        "tables and cause permanent data loss.",
    ),
    _Pattern(
        re.compile(r"\bshred\b"),
        "shred",
        "`shred` overwrites files to make recovery impossible.",
    ),
    _Pattern(
        re.compile(r"\bwipefs\b"),
        "wipefs",
        "`wipefs` erases filesystem signatures, rendering volumes unreadable.",
    ),
    _Pattern(
        # curl/wget piped directly into bash/sh/python/ruby/node
        re.compile(
            r"(curl|wget)\b[^|;&\n]*\|[^|;&\n]*(bash|sh|zsh|fish|python[23]?|ruby|node)\b"
        ),
        "pipe-to-shell",
        "Piping a remote URL directly into a shell interpreter can execute "
        "arbitrary untrusted code.",
    ),
    _Pattern(
        re.compile(r":(){ :|:& };:"),
        "fork bomb",
        "This is a classic fork bomb that will exhaust system process resources.",
    ),
    _Pattern(
        # > /dev/sda  or  >> /dev/sda  (raw block device redirect)
        re.compile(r">{1,2}\s*/dev/[sh]d[a-z][0-9]?"),
        "redirect to block device",
        "Redirecting output directly to a raw block device can corrupt disk data.",
    ),
    _Pattern(
        re.compile(r"\bchmod\s+(-[a-zA-Z]*\s+)*777\s+/"),
        "chmod 777 /",
        "`chmod 777 /` (or recursive variants on root) opens every file on the "
        "system to world-read/write/execute.",
    ),
    _Pattern(
        re.compile(r"\bchown\s+.*\s+/\s*$|\bchown\s+.*\s+-R\s+.*\s+/"),
        "chown on /",
        "Recursively changing ownership on `/` can break system-wide permissions.",
    ),
    _Pattern(
        re.compile(r"\btruncate\s+.*-s\s+0\b"),
        "truncate -s 0",
        "`truncate -s 0` silently zeroes file contents with no undo.",
    ),
    _Pattern(
        re.compile(r"\b(poweroff|halt|shutdown)\b"),
        "system shutdown",
        "System shutdown/halt commands will terminate all running processes and "
        "the operating system itself.",
    ),
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _is_destructive(command: str) -> _Pattern | None:
    """Return the first matching _Pattern, or None if the command is safe."""
    for pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.regex.search(command):
            return pattern
    return None


def _build_block_response(pattern: _Pattern, command: str) -> dict:
    return {
        "decision": "block",
        "reason": (
            f"[Destructive Bash Guard] Blocked — matched rule '{pattern.label}'.\n\n"
            f"{pattern.explanation}\n\n"
            f"Offending command:\n  {command}\n\n"
            "If you are certain this is safe, run the command manually in your "
            "terminal outside of Claude Code."
        ),
    }


def _build_approve_response() -> dict:
    return {"decision": "approve"}


def main() -> None:
    raw = sys.stdin.read()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Cannot parse → approve so we never silently block legitimate work
        # due to malformed hook input.
        print(json.dumps(_build_approve_response()))
        return

    # The tool arguments live in ``payload["input"]`` per the Claude Code
    # hook protocol.  Guard against missing/unexpected shapes.
    tool_input = payload.get("input") or {}
    command: str = tool_input.get("command", "")

    if not isinstance(command, str):
        print(json.dumps(_build_approve_response()))
        return

    matched = _is_destructive(command)
    if matched:
        print(json.dumps(_build_block_response(matched, command)))
    else:
        print(json.dumps(_build_approve_response()))


if __name__ == "__main__":
    main()
