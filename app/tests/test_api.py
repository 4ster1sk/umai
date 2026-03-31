from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_DATA = {
    "おやつ": [
        {"date": "2025-10-16", "item": "赤福"},
        {"date": "2025-11-22", "item": "福砂屋 カステラ"},
    ]
}


@pytest.fixture
def client():
    return TestClient(app)


def test_index_returns_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_api_data_shape(client):
    with patch("app.main.load_data", return_value=SAMPLE_DATA):
        res = client.get("/api/data")
    assert res.status_code == 200
    body = res.json()
    assert "provider" in body
    assert "model" in body
    assert "categories" in body
    assert "おやつ" in body["categories"]
    assert "赤福" in body["categories"]["おやつ"]


def test_api_analyze_streams(client):
    def fake_stream(_data):
        yield "分析"
        yield "結果"

    with patch("app.main.load_data", return_value=SAMPLE_DATA), \
         patch("app.main.stream_analyze", side_effect=fake_stream):
        res = client.post("/api/analyze")

    assert res.status_code == 200
    assert res.text == "分析結果"
