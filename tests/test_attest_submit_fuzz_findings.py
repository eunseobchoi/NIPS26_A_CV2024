import unittest
import json
import sys
import os

# Dynamically add the project root to sys.path to allow imports from 'node'.
# This assumes 'tests' is at the project root level, and 'node' is also at the root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# The original issue describes the file as 'node/rustchain_v2_integrated_v2.2.1_rip200.py'.
# For standard Python module imports, dots in filenames are typically replaced with underscores.
# Assuming the actual importable module name follows this convention:
from node.rustchain_v2_integrated_v2_2_1_rip200 import app

class TestAttestSubmitFuzzFindings(unittest.TestCase):
    """
    Tests derived from the rustchain-1112 fuzzing campaign.
    These tests ensure that the observed correct error handling for
    specific adversarial inputs is maintained as a regression safeguard.
    """
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_attest_submit_invalid_root_null_json(self):
        """
        Verifies the /attest/submit endpoint correctly handles a 'null'
        JSON payload as the root, returning HTTP 400 BAD REQUEST
        with 'INVALID_JSON_OBJECT' error. This scenario was reported
        by the local fuzzing campaign rustchain-1112.
        """
        response = self.app.post('/attest/submit',
                                 data='null',
                                 content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('application/json', response.headers.get('Content-Type'))
        
        response_data = json.loads(response.data)
        self.assertFalse(response_data.get('ok', True), "Expected 'ok': false in response for malformed JSON")
        self.assertEqual(response_data.get('error'), 'INVALID_JSON_OBJECT')

    def test_attest_submit_wrong_types_signature(self):
        """
        Verifies the /attest/submit endpoint correctly handles a payload
        where the 'signature' field is of an incorrect type (e.g., an integer
        instead of a string), returning HTTP 400 BAD REQUEST with
        'INVALID_SIGNATURE_TYPE' error. This behavior was reported
        by the local fuzzing campaign rustchain-1112.
        """
        malformed_payload = {
            "public_key": "dummy_public_key_hex_string",
            "signature": 12345, # Fuzzed input: integer, expected string
            "message": "dummy_message_string",
            "nonce": 1
        }
        response = self.app.post('/attest/submit', json=malformed_payload)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('application/json', response.headers.get('Content-Type'))
        
        response_data = json.loads(response.data)
        self.assertFalse(response_data.get('ok', True), "Expected 'ok': false in response for wrong signature type")
        self.assertEqual(response_data.get('error'), 'INVALID_SIGNATURE_TYPE')

if __name__ == '__main__':
    unittest.main()
