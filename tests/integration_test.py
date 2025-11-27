"""Integration-style test that calls the FastAPI app in-process using TestClient.

This does not require uvicorn or network access and runs the `/predict` endpoint
in mock mode by setting the environment variable `MOCK_PREDICTION`.
"""
import os
from fastapi.testclient import TestClient
from backend.main import app


def main():
    os.environ['MOCK_PREDICTION'] = '1'
    client = TestClient(app)

    # send a minimal fake image payload (empty file) — core will return mock
    files = {'file': ('test.jpg', b'', 'image/jpeg')}
    resp = client.post('/predict', files=files)
    print('Status:', resp.status_code)
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


if __name__ == '__main__':
    main()
