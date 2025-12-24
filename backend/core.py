"""Core image analysis logic.

Uses trained PyTorch model for facial predictions (Age, Gender, Race).
"""
import io
import time
import traceback
from PIL import Image
from typing import Any, Dict

# Import the PyTorch predictor
try:
    from backend.inference_pytorch import FacialPredictor
except ImportError:
    # Fallback for when running from root
    from inference_pytorch import FacialPredictor

# Global model cache
_predictor = None

def _get_predictor():
    """Load the PyTorch predictor (cached)"""
    global _predictor
    if _predictor is None:
        print("Loading PyTorch model...")
        _predictor = FacialPredictor(model_path='models/best_model.pth')
    return _predictor

def analyze_image(image_bytes: bytes, use_mock: bool = False) -> Dict[str, Any]:
    """Analyze an image given as bytes and return predictions.
    
    Args:
        image_bytes: Raw image bytes
        use_mock: If True, returns mock data (ignored for now as we want real inference)
    """
    start_time = time.time()
    
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        width, height = img.size
        
        # Get predictor
        predictor = _get_predictor()
        
        # Predict
        results = predictor.predict(img)
        
        # Format results for frontend
        faces = []
        for res in results:
            # Map inference result to frontend expected format
            x, y, w, h = res['box']
            
            # Ensure race string is capitalized properly for display
            race_display = res['race'].title()
            
            faces.append({
                'face_location': {'x': x, 'y': y, 'w': w, 'h': h},
                'age': res['age'],
                'gender': res['gender'],
                'dominant_race': race_display,
                'gender_confidence': float(res['gender_conf']) * 100,
                'race_scores': res['race_scores'], # Already dict
                'emotion_scores': res['emotion_scores'],
                'quality': {'score': 95} # Placeholder for now
            })
            
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'processing_time_ms': processing_time,
            'metadata': {
                'image_dimensions': {'width': width, 'height': height}
            },
            'faces': faces
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

