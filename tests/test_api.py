"""FastAPI endpoint tests."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_segments_and_filtered_customers() -> None:
    segments = client.get("/v1/segments")
    assert segments.status_code == 200
    segment_id = segments.json()["segments"][0]["segment_id"]
    customers = client.get("/v1/customers", params={"segment_id": segment_id, "limit": 10})
    assert customers.status_code == 200
    assert customers.json()["customers"]
    assert all(row["segment_id"] == segment_id for row in customers.json()["customers"])


def test_real_time_segmentation() -> None:
    request_path = Path(__file__).resolve().parents[1] / "docs" / "example_segment_request.json"
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    response = client.post("/v1/segment", json=payload)
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["segment_id"] >= 1
    assert result["segment_name"]


def test_validation_error() -> None:
    response = client.post("/v1/segment", json={"customers": [{}]})
    assert response.status_code == 422
