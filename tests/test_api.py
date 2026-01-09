from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app": "ai-vision-pipeline"}

def test_detection_endpoint_invalid_file():
    # Sending text file instead of image
    response = client.post(
        "/api/v1/detection/predict",
        files={"file": ("test.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == 400

def test_ocr_endpoint_invalid_file():
    # Sending text file instead of image
    response = client.post(
        "/api/v1/ocr/extract",
        files={"file": ("test.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == 400

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
