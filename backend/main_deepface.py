from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

from backend.core_deepface import analyze_image

app = FastAPI(title="AI Facial Insights API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint for Render.com"""
    return {"status": "healthy", "service": "AI Facial Insights API", "version": "1.0.0"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accept an uploaded image, run analysis with DeepFace and return predictions."""
    try:
        contents = await file.read()
        result = analyze_image(contents, use_mock=False)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)
