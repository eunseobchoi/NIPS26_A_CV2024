```python
import re

with open('README.md', 'r') as f:
    content = f.read()

# Fix truncated line
content = content.replace('run t\n', 'run the following command:\n')

# Add security patch section
security_section = """
## Security Patch (UPLC-CAPE #187)

This repository includes a security fix that rejects submission-local `cape-tests.json` files during verification and measurement.  
See [patch artifact](https://github.com/eunseobchoi/NIPS26_A_CV2024/blob/main/bounty-artifacts/uplc-cape-187/0001-Keep-submission-tests-from-overriding-scenario-contr.patch) for details.
"""

if '## Security Patch' not in content:
    content += security_section

with open('README.md', 'w') as f:
    f.write(content)
```