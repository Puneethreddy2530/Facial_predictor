# AI Facial Insights 

Real-time facial analysis powered by DeepFace CNN models with a modern glassmorphism UI.

[![Live Demo](https://img.shields.io/badge/Demo-GitHub_Pages-blue)](https://puneethreddy2530.github.io/Facial_predictor/)
[![GitHub](https://img.shields.io/badge/GitHub-Puneethreddy2530%2FFacial__predictor-181717?logo=github)](https://github.com/Puneethreddy2530/Facial_predictor)

## Features 

- **Age Detection**: Predicts age with 3-5 year accuracy, categorized into Infant/Child/Teen/Adult/Senior bands
- **Gender Classification**: 97% accuracy using pre-trained VGG-Face/FaceNet models
- **Emotion Analysis**: 7 categories (angry, disgust, fear, happy, sad, surprise, neutral)
- **Race Detection**: 6 categories (asian, indian, black, white, middle eastern, latino hispanic)
- **Quality Checks**: Blur detection and face-size validation to avoid low-quality predictions
- **Confidence Gating**: Flags uncertain emotion/race results when max probability < 40%
- **Largest-Face Cropping**: Automatically detects and analyzes the largest face using RetinaFace
- **Glassmorphism UI**: Modern, responsive frontend with card-based results and loader
- **GitHub Pages Ready**: Static frontend can call any backend via configurable API URL

## Live Demo 

Frontend is hosted on GitHub Pages: [https://puneethreddy2530.github.io/Facial_predictor/](https://puneethreddy2530.github.io/Facial_predictor/)

Configure the backend API in the top-right corner or via ?api=https://your-backend.com.

## Tech Stack 

- **Backend**: FastAPI + DeepFace (TensorFlow 2.20, VGG-Face, RetinaFace)
- **Frontend**: HTML/CSS/JS (glassmorphism design, responsive grid)
- **Models**: Pre-trained CNNs (VGG-Face, FaceNet, ArcFace for embeddings)
- **Quality**: Blur detection (Laplacian variance), face size gating, confidence thresholds
- **Deployment**: Docker, GitHub Pages, GitHub Actions CI/CD

## Quick Start (Windows PowerShell) 

### Backend (Python 3.11 + DeepFace)

1. **Clone the repo**
\\\powershell
git clone https://github.com/Puneethreddy2530/Facial_predictor.git
cd Facial_predictor
\\\

2. **Install Python 3.11** from [python.org](https://www.python.org/downloads/) or run:
\\\powershell
.\install_python311.ps1
\\\

3. **Activate environment and install dependencies**
\\\powershell
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
pip install deepface
\\\

4. **Start backend**
\\\powershell
python -m uvicorn backend.main_deepface:app --host 0.0.0.0 --port 8001
\\\

Backend will listen on http://localhost:8001. First prediction downloads models (~300MB, 10-30s delay).

### Frontend

\\\powershell
cd frontend
python -m http.server 3000
\\\

Open http://localhost:3000 and set API to http://localhost:8001 in the top-right.

## GitHub Pages Deployment 

The frontend is auto-deployed via GitHub Actions on every push to main.

**Manual setup**:
1. Go to repo **Settings  Pages**
2. Source: **GitHub Actions**
3. Workflow: .github/workflows/deploy-pages.yml (already configured)
4. Site URL: https://<username>.github.io/Facial_predictor/

**Backend hosting options**:
- **Render.com**: Free tier, deploy ackend/main_deepface.py as web service
- **Railway.app**: Auto-deploy from GitHub with Dockerfile
- **Azure App Service**: Python 3.11, Linux container
- **Google Cloud Run**: Serverless, pay-per-request

Update the frontend API URL to point to your deployed backend.

## Project Structure 

\\\
Facial_predictor/
 backend/
    core_deepface.py       # DeepFace analysis with quality checks
    main_deepface.py        # FastAPI app
    core.py                 # Legacy Random Forest models
 frontend/
    index.html              # Glassmorphism UI
    styles.css              # Responsive grid + cards
    script.js               # Fetch + quality warnings
    404.html                # GH Pages redirect
 .github/workflows/
    deploy-pages.yml        # Auto-deploy to Pages
 training/
    train_sklearn_model.py # Random Forest training
    visualize_sklearn_results.py
 data_collection/
    generate_synthetic_data.py
 tests/
    test_core.py
    integration_test.py
 Dockerfile                  # Backend containerization
 docker-compose.yml          # Multi-service setup
 README.md
\\\

## API Documentation 

**POST /predict**

Multipart form-data with ile field (image).

Response:
\\\json
{
  "age": 28,
  "age_band": "Adult",
  "gender": "Man",
  "dominant_emotion": "happy",
  "emotion": {
    "angry": 2.1,
    "disgust": 0.01,
    "fear": 1.3,
    "happy": 78.5,
    "sad": 4.2,
    "surprise": 3.9,
    "neutral": 10.0
  },
  "dominant_race": "asian",
  "race": {
    "asian": 62.0,
    "indian": 8.3,
    "black": 1.6,
    "white": 10.5,
    "middle eastern": 2.5,
    "latino hispanic": 15.0
  },
  "low_quality": false,
  "quality_reason": "",
  "low_confidence": false,
  "facial_area": {"x": 123, "y": 45, "w": 200, "h": 230}
}
\\\

## Training (Optional) 

Random Forest models trained on UTKFace (23,708 images):
\\\powershell
python training/train_sklearn_model.py
python training/visualize_sklearn_results.py
\\\

Results: Age MAE 10.01 years, Gender 81.6%, Race 65.4%. DeepFace CNNs significantly outperform (95%+ accuracy).

## Docker 

\\\ash
docker-compose up --build
\\\

Services: backend (8001), nginx (80). Frontend served via nginx.

## Contributing 

1. Fork the repo
2. Create feature branch: git checkout -b feature/amazing-feature
3. Commit: git commit -m 'Add amazing feature'
4. Push: git push origin feature/amazing-feature
5. Open Pull Request

## License 

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments 

- [DeepFace](https://github.com/serengil/deepface) for pre-trained models
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/) for training data
- [FastAPI](https://fastapi.tiangolo.com/) for backend framework

---

Made with  by [Puneethreddy2530](https://github.com/Puneethreddy2530)
