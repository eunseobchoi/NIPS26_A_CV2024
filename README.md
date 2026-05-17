```python
import os
import re

README_PATH = "README.md"
BOUNTY_SECTION_HEADER = "## Bounty Artifact: Human-Connection #1832"
BOUNTY_CONTENT = """\
## Bounty Artifact: Human-Connection #1832

A security patch artifact for [Human-Connection/Human-Connection#1832](https://github.com/Human-Connection/Human-Connection/issues/1832) is hosted in this repository under `bounty-artifacts/human-connection-1832/`.  
The patch prevents email existence disclosure via GraphQL error messages during signup and email-change flows.

- **Patch file:** [0001-avoid-disclosing-whether-submitted-emails-already-exist.patch](bounty-artifacts/human-connection-1832/0001-avoid-disclosing-whether-submitted-emails-already-exist.patch)
- **Artifact commit:** `809bf7f191ee5cba2b3870bf8f6d4c985fd9d0f6`
- **Local patch commit (prepared for upstream):** `f1437107894a8859a5e3c45d1dec7303aa561ac9`

The fix ensures that public mutation responses remain generic when a submitted email is already claimed, removing the account-exists `UserInputError` and sanitizing internal responses.
"""

def main():
    if not os.path.exists(README_PATH):
        raise FileNotFoundError(f"{README_PATH} not found")

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if the section already exists to avoid duplication
    if BOUNTY_SECTION_HEADER in content:
        print("Bounty section already present, no changes made.")
        return

    # Append the bounty section at the end of the file
    # Ensure there's a newline separation if the file doesn't end with one
    if not content.endswith("\n"):
        content += "\n"
    content += "\n" + BOUNTY_CONTENT

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("Bounty section added to README.md")

if __name__ == "__main__":
    main()
```