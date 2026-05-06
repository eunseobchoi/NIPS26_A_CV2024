This directory contains the exact script versions used to generate the
released result JSONs. Each result JSON records the SHA-256 of the
script that produced it; the matching script is preserved here, with
absolute paths replaced by paths relative to the repository root or
`$CAPSULE_ROOT`. `checksums.txt` covers both the rerunnable copies and
the byte-frozen provenance copies.

We store these as regular files rather than symlinks so that zip
extraction, archival mirrors, and checksum verification produce
identical bytes on every system, including those that do not preserve
symlink metadata. The rerunnable copy
`src/counterfactual/phase5_counterfactual_v5_snapshot.py` is a
convenient entry point; its provenance counterpart is
`phase5_counterfactual_v5_be3fd06_exact.py`.

Result JSONs and CSVs use repository-relative paths so that their bytes
are identical regardless of where the repository is checked out. The
script SHA-256 values remain inside each result file; the matching
script is preserved here and listed in the table below.

Some preserved docstrings use the earlier shorthand "video-level
split". The terminology in the current paper supersedes this: we use
the official Kvasir-Capsule split CSVs exactly as released, but a
filename-prefix audit finds 7 shared video prefixes across split_0 and
split_1, so we treat the splits as official frame-list folds rather
than video-disjoint evidence.

| Packaged file | Execution SHA-256 in result JSON | Packaged portable SHA-256 |
| --- | --- | --- |
| `phase5_counterfactual_v5_be3fd06.py` | `be3fd06fab1a04d7311f70ab9f1eb0563a7c8f1db5b2c6081987e04e68b90edf` | `0199652d00c49a5d83d12728bac0e1c42bb44c236917622b4231fd96408ff71e` |
| `phase5_counterfactual_v4_72047d35.py` | `72047d35682f487375235458375f6359ef9d4ba00c94d9fb5aaf0c7fd1237e5c` | `46a6cc2c82f5db668b9bd79c78c1acdad58bcbd68de56e834978b2b9d82f9d88` |
| `phase5_exp3_mi_probe_97312ac3.py` | `97312ac3f5f87816a5f207b57ed98f82d041defff85dbbae97bfd8224eb77af9` | `c084c9190ec2ade08e44b3023383aee87de52721afe16ccbdb4e08195bc112eb` |
| `phase5_exp2b_split0_only_c06c5693.py` | `c06c5693d2d09ed65e6419f6a29107c6f1ef159638d4d291f40ed1c9e9155137` | `20343d3aba8ab719ed3fa99d608cf8621f4816b0fb165264357eeeedd2707875` |

Exact-copy reference versions:

| Provenance file | Historical execution SHA-256 | Packaged anonymized SHA-256 |
| --- | --- | --- |
| `phase5_counterfactual_v5_be3fd06_exact.py` | `be3fd06fab1a04d7311f70ab9f1eb0563a7c8f1db5b2c6081987e04e68b90edf` | `0199652d00c49a5d83d12728bac0e1c42bb44c236917622b4231fd96408ff71e` |
| `phase5_counterfactual_v5_auc_09fa4278_exact.py` | `09fa4278acbbaef8258beae4dbb2f7ea4a6c897a77d8ddbc1b531b8d9fd0ec86` | `09fa4278acbbaef8258beae4dbb2f7ea4a6c897a77d8ddbc1b531b8d9fd0ec86` |
| `phase5_counterfactual_v5_71c2399e_exact.py` | `71c2399e6c9ab91d754ff70cc525ada14083b00fb6ddace876c3ff65cbc4ef1f` | `71c2399e6c9ab91d754ff70cc525ada14083b00fb6ddace876c3ff65cbc4ef1f` |
| `phase5_counterfactual_v5_16f3d70d_exact.py` | `16f3d70d441e23a76100a9f23518e85221c5cb27bcb4fe58f52ad4ff13d0bb7d` | `16f3d70d441e23a76100a9f23518e85221c5cb27bcb4fe58f52ad4ff13d0bb7d` |
| `phase5_counterfactual_v4_72047d35_exact.py` | `72047d35682f487375235458375f6359ef9d4ba00c94d9fb5aaf0c7fd1237e5c` | `46a6cc2c82f5db668b9bd79c78c1acdad58bcbd68de56e834978b2b9d82f9d88` |
| `04_official_test_eval_f834a5_exact.py` | `f834a5bcce8fdf5e8462af690df3d1d0b30d6ca04cb01fdd737749b449a1cfa1` | `f834a5bcce8fdf5e8462af690df3d1d0b30d6ca04cb01fdd737749b449a1cfa1` |
| `phase5_exp3_mi_probe_97312ac3_exact.py` | `97312ac3f5f87816a5f207b57ed98f82d041defff85dbbae97bfd8224eb77af9` | `c084c9190ec2ade08e44b3023383aee87de52721afe16ccbdb4e08195bc112eb` |
| `phase5_exp2b_split0_only_c06c5693_exact.py` | `c06c5693d2d09ed65e6419f6a29107c6f1ef159638d4d291f40ed1c9e9155137` | `70bd6378938d604fcfd4dabb4ee41bbeb671f688ae484cce0e09c09bb1e3011f` |

Same-source/domain re-exposure and Comp-C strengthening JSON files with
`script_sha256=71c2399e6c9ab91d754ff70cc525ada14083b00fb6ddace876c3ff65cbc4ef1f`
are preserved byte-for-byte as
`src/provenance/phase5_counterfactual_v5_71c2399e_exact.py`. The active
`src/counterfactual/phase5_counterfactual_v5.py` is the same rerun surface
with portable same-source/domain path resolution added.

The AUC audit and LOSO result JSONs record
`script_sha256=09fa4278acbbaef8258beae4dbb2f7ea4a6c897a77d8ddbc1b531b8d9fd0ec86`.
That exact execution snapshot is preserved as
`src/provenance/phase5_counterfactual_v5_auc_09fa4278_exact.py`; the active
rerunnable `src/counterfactual/phase5_counterfactual_v5.py` keeps the same
analysis logic but uses the release-package path helpers.

Two Comp-D merged result JSONs record
`script_sha256=16f3d70d441e23a76100a9f23518e85221c5cb27bcb4fe58f52ad4ff13d0bb7d`.
The matching exact execution snapshot is bundled as
`src/provenance/phase5_counterfactual_v5_16f3d70d_exact.py` (MD5
`a4214f030aa621a11cc7cb84b263329c`). The Comp-D training CSVs, per-seed
logs embedded in the merged JSONs, merge manifest, and current rerunnable v5
script are also packaged.
The per-seed Comp-D shard JSONs referenced by the merge manifests are bundled
under `results/strengthening/compD*.json`.

One frozen label-shuffle result records
`6c47bf1531b9d90bdaa31f4cd3feeb42b2c836d94dfe06933121803a20de8ccc`;
the launcher script was not preserved, but the result JSON, the
configuration, and a rerunnable non-frozen label-shuffle script are
included.

One AIIMS-test baseline file used only for the consistency check
records `188f7b4a5a6dc4e746f29a2f87a19db5afc2742dd3d6aeb286d548573c3b3995`.
It is not used for a paper table; the n=10 AIIMS-test direct
evaluation uses the included `04_official_test_eval_f834a5_exact.py`.
