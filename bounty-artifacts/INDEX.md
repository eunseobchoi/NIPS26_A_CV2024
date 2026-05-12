# Public Artifact Index

Generated: `2026-05-12T10:37:16Z`

Purpose: central index of public, non-sensitive bounty artifacts created while direct upstream PRs, private reports, and target issue comments are blocked by missing GitHub user authentication or target repository restrictions.

This file is not acceptance or payout evidence.

## Public Artifacts

| Candidate | Upstream reference | Public artifact | Status |
| --- | --- | --- | --- |
| RustChain #398 | `Scottcjn/rustchain-bounties#398` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/1 | Public Step 1/2 artifact and maintainer visibility comment; private Step 3 still needs GitHub Security Advisory auth. |
| RustChain #73 / PR #4763 | `Scottcjn/rustchain-bounties#73`, `Scottcjn/Rustchain#4763` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/2 | Public review artifact; direct PR review blocked. |
| RustChain #73 / PR #4769 | `Scottcjn/rustchain-bounties#73`, `Scottcjn/Rustchain#4769` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/3 | Public review artifact; direct PR review blocked. |
| RustChain #2819 | `Scottcjn/rustchain-bounties#2819` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/4 | Public report and patch artifact; upstream PR blocked. |
| RustChain #66 | `Scottcjn/rustchain-bounties#66` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/5 | Redacted public availability artifact; full details intentionally not publicized outside target bounty thread. |
| RustChain #1112 | `Scottcjn/rustchain-bounties#1112` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/6 | Public fuzzing proof and report artifact. |
| Doichain #116 | `Doichain/dapp#116` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/7 | Public patch artifact; target repositories are archived/read-only. |
| UPLC-CAPE #187 | `IntersectMBO/UPLC-CAPE#187` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/8 | Public patch artifact; upstream PR blocked. |
| Claude Builders #3 | `claude-builders-bounty/claude-builders-bounty#3` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/9 | Public patch artifact only; required `/opire try` and upstream PR still blocked. |
| Human-Connection #1832 | `Human-Connection/Human-Connection#1832` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/10 | Public patch artifact; upstream PR blocked. |
| Archestra #3854 | `archestra-ai/archestra#3854` | https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/11 | Public implementation report; upstream PR/comment blocked. |

## Current Blockers

- GitHub CLI is unauthenticated.
- `GH_TOKEN` and `GITHUB_TOKEN` are unset.
- No existing SSH forks exist for the checked target repositories under `samowl`.
- GitHub connector writes work for `eunseobchoi/NIPS26_A_CV2024`, but target repository issue comments and PRs fail with `403 Resource not accessible by integration` or repository interaction restrictions.
- Private advisory submissions for Capgo, Librarfree, and RustChain #398 still require authenticated GitHub Security Advisory access.
- At least one maintainer acceptance and one payout receipt are still required before the thread goal can be complete.
