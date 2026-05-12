# RustChain #1112 /attest/submit Fuzzing Report

Date: 2026-05-12
Target: local Flask test client for `node/rustchain_v2_integrated_v2.2.1_rip200.py`
Issue: https://github.com/Scottcjn/rustchain-bounties/issues/1112

## Summary

- Total malformed/adversarial payloads tested: 160
- Status counts: `{"200": 3, "400": 56, "409": 96, "422": 5}`
- Server-side 500-class failures: 0
- Slow cases >= 1000 ms: 0
- Maximum observed handler time: 191.82 ms

## Category Outcomes

| Category | Response status counts |
| --- | --- |
| fingerprint_edge | `{"409": 10}` |
| injection_style | `{"200": 1, "409": 29}` |
| malformed_json | `{"400": 4}` |
| missing_fields | `{"200": 2, "400": 2, "409": 23}` |
| nonce_edge | `{"400": 6, "409": 11}` |
| oversized_inputs | `{"409": 23}` |
| signature_edge | `{"400": 13}` |
| wrong_types | `{"400": 31, "422": 5}` |

## Top Error Codes

```json
{
  "CHALLENGE_INVALID": 53,
  "INVALID_DEVICE": 5,
  "INVALID_DEVICE_CORES": 5,
  "INVALID_FINGERPRINT": 1,
  "INVALID_FINGERPRINT_CHECKS": 2,
  "INVALID_JSON_OBJECT": 4,
  "INVALID_MINER": 5,
  "INVALID_PUBLIC_KEY_TYPE": 1,
  "INVALID_REPORT": 7,
  "INVALID_SIGNALS": 3,
  "INVALID_SIGNALS_MACS": 5,
  "INVALID_SIGNATURE": 13,
  "INVALID_SIGNATURE_TYPE": 2,
  "MISSING_NONCE": 8,
  "REPLAY_ATTACK_BLOCKED": 43
}
```

## Notable Reproduction Cases

- `invalid_root_null` (malformed_json): HTTP 400, code `INVALID_JSON_OBJECT`, 0.49 ms; request preview `null`
- `wrong_types_signature` (wrong_types): HTTP 400, code `INVALID_SIGNATURE_TYPE`, 0.19 ms; request preview `{"device": {"cores": 8, "cpu": "IBM POWER8", "device_arch": "power8", "device_family": "PowerPC", "serial_number": "SERIAL-0018"}, "fingerprint": {"checks": {"anti_emulation": {"da`
- `injection_miner_38` (injection_style): HTTP 409, code `CHALLENGE_INVALID`, 0.61 ms; request preview `{"device": {"cores": 8, "cpu": "IBM POWER8", "device_arch": "power8", "device_family": "PowerPC", "serial_number": "SERIAL-0038"}, "fingerprint": {"checks": {"anti_emulation": {"da`
- `oversized_signals_macs` (oversized_inputs): HTTP 409, code `REPLAY_ATTACK_BLOCKED`, 191.82 ms; request preview `{"device": {"cores": 8, "cpu": "IBM POWER8", "device_arch": "power8", "device_family": "PowerPC", "serial_number": "SERIAL-0054"}, "fingerprint": {"checks": {"anti_emulation": {"da`

## Findings

The campaign did not find an unhandled exception or endpoint crash. Malformed payload shapes were rejected before normalization with 400/422 responses, replay or invalid nonce paths returned 409, and invalid Ed25519 signature material returned a non-500 error path.

The highest-cost tested input class was oversized metadata and MAC arrays. In this local run those cases completed below the 1000 ms slow-case threshold, so I do not have evidence of an endpoint-level DoS from the tested payload sizes.

## Reproduction

```bash
uv run --no-project --with flask --with pynacl python \
  /home/user/bounty-submissions/rustchain-1112/attest_submit_fuzz_campaign.py \
  --repo /home/user/Rustchain-bounty-1112 \
  --cases 160 \
  --seed 1112
```

The raw JSON result file includes each payload category, request preview, HTTP status, response code, response preview, and elapsed handler time.

No production RustChain node was probed; all traffic used Flask `app.test_client()` locally.