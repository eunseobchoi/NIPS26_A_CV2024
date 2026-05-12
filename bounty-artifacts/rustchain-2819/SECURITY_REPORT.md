# RustChain #2819 UTXO Red-Team Finding: Malformed Field Validation Hardening

Date: 2026-05-12
Target: RustChain `node/utxo_endpoints.py` and `node/utxo_db.py` (Flask + SQLite)
Issue: https://github.com/Scottcjn/rustchain-bounties/issues/2819
Patch: `0001-reject-malformed-utxo-transfer-fields.patch`

## Summary

Two fail-closed gaps were found and fixed in the UTXO transfer endpoint and mempool/state-transition validation:

1. **Endpoint field type confusion** (`utxo_endpoints.py`): The `/utxo/transfer` handler called `.strip()` on `from_address`, `to_address`, `public_key`, and `signature` before verifying those fields were strings. A payload such as `"from_address": ["not", "string"]` raised an unhandled `AttributeError`, producing a 500 response.

2. **Mempool output address gap** (`utxo_db.py`): `UtxoDB.mempool_add()` checked that output `value_nrtc` was a positive integer but did not validate that the output `address` was present and a non-empty string. A transaction with well-formed values but a missing or non-string `address` could enter the mempool even though `apply_transaction()` would later be unable to derive output box propositions for block inclusion, creating an unmineable candidate that locks input UTXOs until mempool expiry.

## Impact

| Criterion | Rating | Rationale |
|-----------|--------|-----------|
| Severity | Low–Medium (#2819 payout table) | No fund loss; temporary UTXO lock + public 500 on unauthenticated endpoint |
| Attack vector | Public unauthenticated HTTP / DoS | Malformed JSON triggers 500; malformed mempool payload locks UTXOs until expiry |
| Remediation cost | Low | 4-line validator function + 2 integration checks + 1 boundary alignment |

## Changes

### `node/utxo_endpoints.py` (26 insertions, 4 deletions)

- Checks that the JSON root is a `dict` before processing; non-object JSON returns `400/INVALID_JSON_OBJECT`.
- Introduces `_optional_text_field()` which first asserts `isinstance(raw, str)` before calling `.strip()`. Non-string values raise `ValueError` caught by the endpoint and returned as `400/INVALID_FIELD_TYPE`.
- Replaces four inline `(data.get(...) or '').strip()` calls with the bounded validator.

### `node/utxo_db.py` (27 insertions, 3 deletions)

- Adds `_valid_output_address()`: returns `True` iff the value is a non-empty string.
- In `apply_transaction()`: validates each output is a `dict`, validates address before processing, and switches from `sum(o['value_nrtc'] for o in outputs)` to an explicit loop that sums only after all per-output checks pass.
- In `mempool_add()`: validates each output is a `dict` and that every output carries a valid address, matching the `apply_transaction()` validation.

### `node/test_utxo_db.py` (65 new lines)

| Test | What it covers |
|------|----------------|
| `test_transfer_rejects_missing_output_address` | Outputs without an `address` key are rejected by `apply_transaction()` |
| `test_transfer_rejects_non_string_output_address` | Outputs with a non-string `address` (e.g. `["bob"]`) are rejected by `apply_transaction()` |
| `test_mempool_rejects_missing_output_address` | Mempool rejects and does not lock UTXOs when outputs lack an `address` |
| `test_mempool_rejects_non_string_output_address` | Mempool rejects and does not lock UTXOs when outputs have a non-string `address` |

### `node/test_utxo_endpoints.py` (36 new lines)

| Test | What it covers |
|------|----------------|
| `test_transfer_rejects_non_object_json` | JSON array root (`[]`) returns `400/INVALID_JSON_OBJECT` |
| `test_transfer_rejects_wrong_type_text_fields` | `from_address`, `to_address`, `public_key`, `signature` as non-string types return `400/INVALID_FIELD_TYPE` with field name in error |

## Reproduction

**Endpoint 500 (before fix):**

```python
client.post("/utxo/transfer", json={
    "from_address": ["not", "string"],
    "to_address": "bob",
    "amount_rtc": 1,
    "public_key": "aabbccdd" * 8,
    "signature": "sig",
    "nonce": 1,
})
# → 500 Internal Server Error (AttributeError on .strip())
```

**Expected behavior (after fix):**

```json
{"code": "INVALID_FIELD_TYPE", "error": "from_address must be a string"}
```

**Mempool DoS (before fix):**

```python
db.mempool_add({
    "tx_id": "bad_address_" * 6,
    "inputs": [{"box_id": box["box_id"]}],
    "outputs": [{"address": {"addr": "bob"}, "value_nrtc": 100}],
    "fee_nrtc": 0,
})
# → True (accepted into mempool despite unmineable output)
```

**Expected behavior (after fix):**

```python
# → False (rejected, UTXOs not locked)
```

## Patch Verification

```bash
# Syntax check
python3 -m py_compile node/utxo_db.py node/utxo_endpoints.py \
  node/test_utxo_db.py node/test_utxo_endpoints.py

# Test suite (requires pytest and flask)
uv run --no-project --with pytest --with flask \
  python -m pytest node/test_utxo_db.py node/test_utxo_endpoints.py -q

# Whitespace hygiene
git diff --check
```

## References

- Upstream bounty: https://github.com/Scottcjn/rustchain-bounties/issues/2819
- Public source issue: https://github.com/eunseobchoi/NIPS26_A_CV2024/issues/4
- Patch file: `0001-reject-malformed-utxo-transfer-fields.patch`
