"""
backend/test_api.py
===================
Automated Test Suite for FastAPI Backend.

Tests:
1. GET  /health (Status 200)
2. GET  /model-info (Status 200)
3. GET  /sample-input (Status 200)
4. POST /predict Valid 51-band request (Status 200)
5. POST /predict Invalid - Missing bands (Status 422)
6. POST /predict Invalid - Extra bands > 51 (Status 422)
7. POST /predict Invalid - Non-numeric values (Status 422)
8. POST /predict Invalid - Missing body (Status 422)
"""

import os
import sys
import pandas as pd

# Add backend directory to sys.path
sys.path.append(os.path.dirname(__file__))

from fastapi.testclient import TestClient
from main import app, BAND_COLS

client = TestClient(app)


def test_api_endpoints():
    print("=" * 60)
    print("       FASTAPI BACKEND ENDPOINTS & ERROR HANDLING TEST       ")
    print("=" * 60)

    # 1. Test GET /health
    print("\n1. Testing GET /health...")
    res_health = client.get("/health")
    assert res_health.status_code == 200, f"Expected 200, got {res_health.status_code}"
    print(f"   [OK] Status Code: {res_health.status_code}")
    print(f"   Response: {res_health.json()}")

    # 2. Test GET /model-info
    print("\n2. Testing GET /model-info...")
    res_info = client.get("/model-info")
    assert res_info.status_code == 200, f"Expected 200, got {res_info.status_code}"
    print(f"   [OK] Status Code: {res_info.status_code}")
    print(f"   Architecture: {res_info.json()['model_architecture']}")
    print(f"   Data Source: {res_info.json()['data_source']}")

    # 3. Test GET /sample-input
    print("\n3. Testing GET /sample-input...")
    res_sample = client.get("/sample-input")
    assert res_sample.status_code == 200, f"Expected 200, got {res_sample.status_code}"
    sample_payload = res_sample.json()["payload"]
    print(f"   [OK] Status Code: {res_sample.status_code}")
    print(f"   Sample Bands Count: {len(sample_payload['bands'])}")

    # 4. Test POST /predict VALID Input
    print("\n4. Testing POST /predict (VALID 51 Bands Input)...")
    res_predict = client.post("/predict", json=sample_payload)
    assert res_predict.status_code == 200, f"Expected 200, got {res_predict.status_code}"
    pred_data = res_predict.json()
    print(f"   [OK] Status Code: {res_predict.status_code}")
    print(f"   Predicted Turbidity: {pred_data['predicted_turbidity']} {pred_data['unit']}")
    print(f"   Quality Status:      {pred_data['quality_status']}")
    print(f"   Data Source Tag:     {pred_data['data_source']}")

    # 5. Test POST /predict INVALID: Missing Bands (< 51)
    print("\n5. Testing POST /predict (INVALID: Missing Bands - 10 bands)...")
    incomplete_bands = {f"band_{w}nm": 0.01 for w in range(400, 500, 10)}
    res_inv1 = client.post("/predict", json={"bands": incomplete_bands})
    assert res_inv1.status_code == 422, f"Expected 422, got {res_inv1.status_code}"
    print(f"   [OK] Handled Gracefully! Status Code: {res_inv1.status_code}")
    print(f"   Error Detail: {res_inv1.json()['detail'][0]['msg']}")

    # 6. Test POST /predict INVALID: Extra Bands (> 51)
    print("\n6. Testing POST /predict (INVALID: Extra Bands - 52 bands)...")
    extra_bands = sample_payload["bands"].copy()
    extra_bands["band_910nm"] = 0.05
    res_inv2 = client.post("/predict", json={"bands": extra_bands})
    assert res_inv2.status_code == 422, f"Expected 422, got {res_inv2.status_code}"
    print(f"   [OK] Handled Gracefully! Status Code: {res_inv2.status_code}")
    print(f"   Error Detail: {res_inv2.json()['detail'][0]['msg']}")

    # 7. Test POST /predict INVALID: Non-Numeric Values
    print("\n7. Testing POST /predict (INVALID: Non-Numeric String Value)...")
    string_bands = sample_payload["bands"].copy()
    string_bands["band_400nm"] = "INVALID_STRING"
    res_inv3 = client.post("/predict", json={"bands": string_bands})
    assert res_inv3.status_code == 422, f"Expected 422, got {res_inv3.status_code}"
    print(f"   [OK] Handled Gracefully! Status Code: {res_inv3.status_code}")
    print(f"   Error Detail: {res_inv3.json()['detail'][0]['msg']}")

    # 8. Test POST /predict INVALID: Missing Body
    print("\n8. Testing POST /predict (INVALID: Missing Body / Empty Request)...")
    res_inv4 = client.post("/predict", json={})
    assert res_inv4.status_code == 422, f"Expected 422, got {res_inv4.status_code}"
    print(f"   [OK] Handled Gracefully! Status Code: {res_inv4.status_code}")
    print(f"   Error Detail: {res_inv4.json()['detail'][0]['msg']}")

    # 9. Test GET /stream/next-frame
    print("\n9. Testing GET /stream/next-frame (SIMULATION STREAM)...")
    res_stream = client.get("/stream/next-frame")
    assert res_stream.status_code == 200, f"Expected 200, got {res_stream.status_code}"
    stream_data = res_stream.json()
    print(f"   [OK] Status Code: {res_stream.status_code}")
    print(f"   Frame ID:        {stream_data['frame_id']} at {stream_data['timestamp']}")
    print(f"   Turbidity:       {stream_data['prediction']['predicted_turbidity']} NTU")
    print(f"   Data Source Tag: {stream_data['data_source']}")

    print("\n" + "=" * 60)
    print("      ALL FASTAPI BACKEND TESTS PASSED SUCCESSFULLY!       ")
    print("=" * 60)


if __name__ == "__main__":
    test_api_endpoints()
