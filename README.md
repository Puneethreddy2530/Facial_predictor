# Facial Prediction Prototype

This repository contains a minimal full-stack prototype that analyzes a face image and returns predictions (age, gender, emotion, race) using DeepFace.

Folders:
- `backend/` — FastAPI server providing the `/predict` endpoint.
- `frontend/` — simple HTML/JS UI to upload images and display predictions.

## Requirements

- **For mock mode (testing):** Python 3.10+ (including 3.14)
- **For real ML predictions:** Python 3.10 or 3.11 (TensorFlow not yet available for 3.14)

## Quick start (Windows PowerShell)

1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install minimal dependencies

```powershell
pip install --upgrade pip
pip install fastapi uvicorn[standard] pillow numpy python-multipart httpx
```

3. Run backend in mock mode

```powershell
$env:MOCK_PREDICTION = '1'
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Option B: Real Predictions (Requires Python 3.10 or 3.11)

1. **Install Python 3.11** from https://www.python.org/downloads/

2. **Run automated setup script:**

```powershell
.\setup_py311_env.ps1
```

OR manually:

```powershell
# Use Python 3.11 executable (adjust path as needed)
C:\Python311\python.exe -m venv .venv311
.\.venv311\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
pip install deepface
```

3. Run the backend (real predictions)

```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Note:** Without `MOCK_PREDICTION=1`, the backend will attempt to use DeepFace for real facial analysis.

4. Open the frontend

- Option A: Open `frontend/index.html` directly in your browser.
- Option B: Serve it (recommended) and browse to `http://localhost:3000`:

```powershell
cd frontend

## Docker Deployment

### Quick Start with Docker Compose

# Build and run both frontend and backend
docker-compose up --build

docker-compose up -d

# Stop services
docker-compose down
```

Access the application:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000

```powershell
# Set environment variable for mock predictions
$env:MOCK_PREDICTION='1'
docker-compose up
```

### Build Backend Only

```powershell
# Build the Docker image
docker build -t facial-prediction-backend .

# Run the container
docker run -p 8000:8000 facial-prediction-backend

# Run with mock mode
docker run -p 8000:8000 -e MOCK_PREDICTION=1 facial-prediction-backend
```

### Docker Notes

- Uses Python 3.11 for DeepFace compatibility
- First run downloads ML models (~100-500MB)
- Backend includes health checks
- Nginx serves frontend and proxies API requests

## Notes

- **Python 3.14 Limitation:** TensorFlow (required by DeepFace) is not yet available for Python 3.14. Use Python 3.10 or 3.11 for real predictions.
- **Mock Mode:** Set `$env:MOCK_PREDICTION='1'` to run without ML dependencies (returns sample data).
- **DeepFace Installation:** First run will download pre-trained models (~100-500MB). This is normal.
- **CPU vs GPU:** The default install uses TensorFlow CPU. For GPU acceleration, install `tensorflow-gpu` instead (requires CUDA).

## Testing

Run tests in mock mode (no ML required):

```powershell
python -m tests.test_core
python -m tests.integration_test
```

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py          # FastAPI app with /predict endpoint
│   └── core.py          # Analysis logic (mock + real modes)
├── frontend/
│   ├── index.html       # UI for image upload
│   ├── script.js        # Frontend logic
│   └── styles.css       # Styling
├── tests/
│   ├── test_core.py     # Unit tests (mock mode)
│   └── integration_test.py  # Integration tests
├── requirements.txt     # Python dependencies
├── setup_py311_env.ps1  # Automated setup for Python 3.11
└── README.md
```
