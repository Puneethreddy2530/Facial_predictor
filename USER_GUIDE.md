# 🎭 AI Facial Insights - User Guide

## ✨ For Regular Users (Super Easy!)

### How to Use the App

1. **Visit the website**: [https://puneethreddy2530.github.io/Facial_predictor/](https://puneethreddy2530.github.io/Facial_predictor/)

2. **Upload a photo**: 
   - Click "Choose File" 
   - Select a photo with a clear face
   - You'll see a preview

3. **Click "Analyze"**:
   - Wait a few seconds (first time may take 30-60s)
   - See your results!

4. **View predictions**:
   - **Age**: Estimated age and category (Child/Teen/Adult/etc)
   - **Gender**: Male/Female prediction
   - **Emotion**: Happy, sad, angry, neutral, etc.
   - **Race**: Ethnicity prediction

### ⚡ Tips for Best Results

- ✅ Use **clear, well-lit photos**
- ✅ Face should be **clearly visible** (not too small)
- ✅ Avoid **blurry images**
- ✅ Works with **any device** (phone, tablet, laptop)
- ⚠️ First request takes 30-60s (app wakes up from sleep)
- ⚡ Subsequent requests are fast (2-5 seconds)

### 🤔 Why is the first request slow?

The app uses a **free hosting service (Render.com)** that puts the backend to sleep after 15 minutes of inactivity. When you make the first request, it needs to "wake up" which takes 30-60 seconds. After that, it's fast!

---

## 🔧 For Developers

### Running Locally

If you want to run the app on your own computer:

#### 1. Clone the repository
```bash
git clone https://github.com/Puneethreddy2530/Facial_predictor.git
cd Facial_predictor
```

#### 2. Install Python 3.11+
Download from [python.org](https://www.python.org/downloads/)

#### 3. Create virtual environment and install dependencies
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 4. Run the backend
```powershell
python -m uvicorn backend.main_deepface:app --host 0.0.0.0 --port 8001
```

#### 5. Run the frontend
```powershell
cd frontend
python -m http.server 3000
```

#### 6. Open in browser
- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- API Docs: http://localhost:8001/docs

### Deploying Your Own Backend

Want to deploy your own version? See:
- **Quick Deploy**: [QUICK_DEPLOY.md](QUICK_DEPLOY.md) - 5-minute guide
- **Detailed Guide**: [DEPLOY_GUIDE.md](DEPLOY_GUIDE.md) - Complete instructions

### Project Structure

```
Facial_predictor/
├── frontend/           # HTML/CSS/JS glassmorphism UI
│   ├── index.html
│   ├── styles.css
│   └── script.js
├── backend/            # FastAPI + DeepFace backend
│   ├── main_deepface.py
│   └── core_deepface.py
├── .github/workflows/  # Auto-deployment to GitHub Pages
├── requirements.txt    # Python dependencies
├── render.yaml         # Render.com deployment config
└── README.md
```

### API Endpoints

**Health Check**
```bash
GET https://facial-predictor-api.onrender.com/
```
Returns: `{"status":"healthy","service":"AI Facial Insights API","version":"1.0.0"}`

**Predict**
```bash
POST https://facial-predictor-api.onrender.com/predict
Content-Type: multipart/form-data
Body: file=<image_file>
```
Returns:
```json
{
  "age": 25,
  "age_band": "Adult (18-60)",
  "gender": "Man",
  "dominant_emotion": "happy",
  "emotion": {
    "angry": 0.01,
    "disgust": 0.00,
    "fear": 0.02,
    "happy": 0.92,
    "sad": 0.03,
    "surprise": 0.01,
    "neutral": 0.01
  },
  "dominant_race": "white",
  "race": {
    "asian": 0.05,
    "indian": 0.03,
    "black": 0.02,
    "white": 0.85,
    "middle eastern": 0.03,
    "latino hispanic": 0.02
  },
  "low_quality": false,
  "quality_reason": null,
  "low_confidence": false,
  "facial_area": {"x": 100, "y": 100, "w": 200, "h": 200}
}
```

### Tech Stack

- **Frontend**: Vanilla JavaScript, HTML5, CSS3 (glassmorphism design)
- **Backend**: FastAPI (Python), DeepFace, TensorFlow, OpenCV
- **Models**: VGG-Face, FaceNet, RetinaFace (pre-trained CNNs)
- **Hosting**: 
  - Frontend: GitHub Pages (free)
  - Backend: Render.com (free tier)
- **CI/CD**: GitHub Actions

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Commit: `git commit -m "Add feature-name"`
6. Push: `git push origin feature-name`
7. Open a Pull Request

### License

MIT License - Feel free to use for personal or commercial projects!

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Puneethreddy2530/Facial_predictor/issues)
- **Live Demo**: [https://puneethreddy2530.github.io/Facial_predictor/](https://puneethreddy2530.github.io/Facial_predictor/)
- **API Status**: [https://facial-predictor-api.onrender.com/](https://facial-predictor-api.onrender.com/)

---

**Made with ❤️ using DeepFace and FastAPI**
