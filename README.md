```python
#!/usr/bin/env python3
"""Update README.md with Claude Builders #3 bounty artifact documentation."""

import os

README_PATH = "README.md"
SECTION_TITLE = "## Bounty Artifacts"
ARTIFACT_DESCRIPTION = """\
### Claude Builders #3 — Destructive Bash PreToolUse Hook

This repository includes a patch artifact for the
[Claude Builders Bounty #3](https://github.com/claude-builders-bounty/claude-builders-bounty/issues/3)
that implements a Claude Code `PreToolUse` hook to block destructive Bash
commands before execution.

**Blocked patterns:**
- `rm -rf`
- `DROP TABLE`
- `TRUNCATE`
- `git push --force` / `git push -f`
- `DELETE FROM` without a `WHERE` clause

Blocked attempts are logged to `~/.claude/hooks/blocked.log` with timestamp,
command, project path, and reason.

**Patch file:** [`bounty-artifacts/claude-builders-3/0001-Block-destructive-Bash-commands-before-Claude-tool-u.patch`](bounty-artifacts/claude-builders-3/0001-Block-destructive-Bash-commands-before-Claude-tool-u.patch)  
**Artifact commit:** `0169b567d5e7ce817d60f5ccdc6edc5b2c5657c4`  
**Local patch commit (prepared for upstream):** `ae1ac4c0de6ee32fce8ffda7c60406ce0ca95ffe`

**Verification:**  
- `python3 -m unittest discover -s tests`  
- `./install.sh` with a temporary `HOME`  
- Manual hook invocation with `DELETE FROM users` (blocked)  
- Manual hook invocation with `DELETE FROM users WHERE id = 1` (allowed)

No production probing was performed; this is a local implementation and patch
artifact while direct upstream PR/claim submission is blocked.
"""

def main():
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if SECTION_TITLE in content:
        print(f"Section '{SECTION_TITLE}' already exists. No changes made.")
        return

    # Append the new section at the end of the file
    new_content = content.rstrip() + "\n\n" + SECTION_TITLE + "\n\n" + ARTIFACT_DESCRIPTION + "\n"

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Updated {README_PATH} with bounty artifact section.")

if __name__ == "__main__":
    import sys
    main()
```