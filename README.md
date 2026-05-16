```python
import os

README_PATH = "README.md"
BOUNTY_SECTION = """

## Bounty Artifacts

This repository also hosts implementation artifacts for open bounties.

- **archestra-ai/archestra#3854** ($250 bounty): Admin audit log implementation report → [`bounty-artifacts/archestra-3854/REPORT.md`](bounty-artifacts/archestra-3854/REPORT.md)
"""

def main():
    if not os.path.exists(README_PATH):
        print(f"Error: {README_PATH} not found")
        return

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Avoid duplicate insertion
    if "## Bounty Artifacts" in content:
        print("Bounty section already exists, skipping.")
        return

    # Insert before the Quickstart section or at the end
    quickstart_marker = "## Quickstart"
    if quickstart_marker in content:
        idx = content.index(quickstart_marker)
        new_content = content[:idx] + BOUNTY_SECTION + "\n" + content[idx:]
    else:
        new_content = content.rstrip() + "\n" + BOUNTY_SECTION

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("README.md updated with bounty artifact section.")

if __name__ == "__main__":
    main()
```