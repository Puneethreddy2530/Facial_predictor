"""
Enhanced FastAPI backend with real-time PyTorch inference.
Uses: FastAPI, FaceNet (MTCNN), ResNet34 (Custom), CUDA
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
import io
import time
from PIL import Image, ImageStat
from datetime import datetime
import numpy as np

# Import our new PyTorch predictor
try:
    from backend.inference_pytorch import FacialPredictor
except ImportError:
    # Handle case where run from root
    from inference_pytorch import FacialPredictor

app = FastAPI(
    title="AI Facial Insights PRO (PyTorch Edition)",
    description="Real-time facial analysis using Multi-Task ResNet & CUDA",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    print("🚀 Initializing AI Engines...")
    try:
        predictor = FacialPredictor()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# Response Models
class FaceLocation(BaseModel):
    x: int
    y: int
    w: int
    h: int
    confidence: float

class EmotionScores(BaseModel):
    angry: float = 0.0
    disgust: float = 0.0
    fear: float = 0.0
    happy: float = 0.0
    sad: float = 0.0
    surprise: float = 0.0
    neutral: float = 0.0

class RaceScores(BaseModel):
    white: float
    black: float
    asian: float
    indian: float
    others: float

class FaceQuality(BaseModel):
    score: float
    brightness: float
    sharpness: float

class FacialAnalysis(BaseModel):
    face_id: int
    age: int
    gender: str
    gender_confidence: float
    dominant_emotion: str
    emotion_scores: EmotionScores
    dominant_race: str
    race_scores: RaceScores
    face_location: FaceLocation
    quality: FaceQuality

class PredictionResponse(BaseModel):
    success: bool
    timestamp: str
    processing_time_ms: float
    faces_detected: int
    faces: List[FacialAnalysis]
    warnings: List[str]

    
def analyze_image_quality(img: Image.Image) -> FaceQuality:
    """Analyze image quality metrics"""
    gray = img.convert('L')
    stat = ImageStat.Stat(gray)
    sharpness = min(stat.stddev[0] / 100.0, 1.0) * 100
    brightness = (stat.mean[0] / 255.0) * 100
    
    # Simple composite score
    score = (sharpness * 0.6 + min(abs(brightness - 50) * 2, 100) * 0.4)
    
    return FaceQuality(
        score=round(score, 1),
        brightness=round(brightness, 1),
        sharpness=round(sharpness, 1)
    )

@app.get("/")
async def root():
    return {
        "status": "online",
        "engine": "PyTorch/CUDA",
        "model": "MultiTaskResNet34"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Real-time inference using PyTorch models
    """
    if not predictor:
        raise HTTPException(503, "Model not loaded yet")
        
    start_time = time.time()
    warnings = []
    
    try:
        # 1. Read Image
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception:
            raise HTTPException(400, "Invalid image file")
            
        # 2. Run Inference
        results = predictor.predict(img)
        
        # 3. Format Response
        faces = []
        for i, res in enumerate(results):
            # Quality check
            quality = analyze_image_quality(img.crop((
                res['box'][0], res['box'][1], 
                res['box'][0]+res['box'][2], res['box'][1]+res['box'][3]
            )))
            
            face_loc = FaceLocation(
                x=res['box'][0], y=res['box'][1], 
                w=res['box'][2], h=res['box'][3],
                confidence=res['confidence']
            )
            
            # Map emotion scores
            e_scores = EmotionScores(**res['emotion_scores'])
            r_scores = RaceScores(**res['race_scores'])
            
            faces.append(FacialAnalysis(
                face_id=i,
                age=res['age'],
                gender=res['gender'],
                gender_confidence=round(res['gender_conf'] * 100, 1),
                dominant_emotion=res['emotion'],
                emotion_scores=e_scores,
                dominant_race=res['race'],
                race_scores=r_scores,
                face_location=face_loc,
                quality=quality
            ))
            
        if not faces:
            warnings.append("No faces detected. Try a clearer image.")
            
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            success=True,
            timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=round(processing_time, 2),
            faces_detected=len(faces),
            faces=faces,
            warnings=warnings
        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, str(e))

if __name__ == '__main__':
    import uvicorn
    print("\nStarting PyTorch Backend...")
    uvicorn.run(app, host="127.0.0.1", port=8001)
