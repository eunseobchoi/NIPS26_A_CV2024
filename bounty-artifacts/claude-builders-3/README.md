# Claude Code Destructive Bash Command Block

> A community bounty board for Claude Code builders.

## Bounty #3: Destructive Bash Command Block

This hook prevents Claude Code from executing potentially destructive Bash commands by intercepting them at the PreToolUse stage.

### Blocked Commands

- `rm -rf` (recursive force deletion)
- `git push --force` / `git push -f` (force push)
- `DROP TABLE` (SQL table deletion)
- `TRUNCATE` (SQL table truncation)
- `DELETE FROM` without `WHERE` clause (unqualified deletion)

### Installation

```bash
./install.sh
```

This registers the hook with Claude Code and makes it executable.

### Testing

```bash
python3 -m unittest discover -s tests
```

### Logging

Blocked commands are logged to `~/.claude/hooks/blocked.log` with:
- Timestamp
- Attempted command
- Project path
- Block reason

### Hook Output Format

Follows Claude Code hooks JSON schema for PreToolUse decisions:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Blocked destructive Bash command: <reason>. Command: <command>"
  }
}
```