```python
import os

README_PATH = "README.md"
SECTION_TITLE = "## Bounty Artifacts"

def add_bounty_section():
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found.")
        return

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if SECTION_TITLE in content:
        print("Bounty section already exists. No changes made.")
        return

    bounty_section = """
## Bounty Artifacts

This repository includes a security patch artifact for **Doichain/dapp#116** (bound `sendrawtransaction` body parameters).

- **Patch file:** [`bounty-artifacts/doichain-116/0001-limit-sendrawtransaction-body-parameters.patch`](bounty-artifacts/doichain-116/0001-limit-sendrawtransaction-body-parameters.patch)
- **Commit:** `139ef6c53062ce0869294a7c6a83754369ce8f10`

### Summary

The public `sendrawtransaction` route accepted unbounded request body fields and parsed `params.tx` before size validation. In the DOI branch it could also persist attacker-sized `templateDataEncrypted` values in `OptIns`, creating a denial-of-service risk.

The patch adds configurable byte limits:

- `api.sendrawtransaction.maxTxBytes` (default 100000)
- `api.sendrawtransaction.maxNameIdBytes` (default 512)
- `api.sendrawtransaction.maxTemplateDataEncryptedBytes` (default 65536)

It returns HTTP 400 for missing/non-string fields and HTTP 413 for oversized fields before transaction parsing or database insert.

### Verification

- `git diff --check` passed
- `node --check server/api/rest/imports/send.js` passed
- No production probing was performed; this report is based on local source review and static validation only.
"""

    # Append at the end of the file
    updated_content = content.rstrip() + "\n" + bounty_section

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"Added bounty section to {README_PATH}")

if __name__ == "__main__":
    add_bounty_section()
```