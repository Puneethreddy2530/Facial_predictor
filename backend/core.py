"""Core image analysis logic.

Uses trained Random Forest models for facial predictions.
"""
import io
import os
import cv2
import numpy as np
import joblib
from PIL import Image
from typing import Any, Dict

# Global model cache
_models_cache = {}

def _load_models():
    """Load trained models (cached)"""
    global _models_cache
    
    if not _models_cache:
        models_dir = 'models'
        _models_cache['age'] = joblib.load(os.path.join(models_dir, 'age_model.pkl'))
        _models_cache['gender'] = joblib.load(os.path.join(models_dir, 'gender_model.pkl'))
        _models_cache['race'] = joblib.load(os.path.join(models_dir, 'race_model.pkl'))
        print("✓ Models loaded successfully")
    
    return _models_cache


def analyze_image(image_bytes: bytes, use_mock: bool = False) -> Dict[str, Any]:
    """Analyze an image given as bytes and return predictions.

    Uses trained Random Forest models for age, gender, and race prediction.
    """
    try:
        # Load models
        models = _load_models()
        
        # Convert bytes to image
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        
        # Convert to BGR for OpenCV
        if len(img_array.shape) == 2:  # Grayscale
            img_gray = img_array
        elif img_array.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:  # RGB
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Preprocess for model (same as training)
        img_resized = cv2.resize(img_gray, (64, 64))
        img_flat = img_resized.flatten() / 255.0
        
        # Predict
        age = int(models['age'].predict([img_flat])[0])
        gender_id = int(models['gender'].predict([img_flat])[0])
        race_id = int(models['race'].predict([img_flat])[0])
        
        # Map to labels
        gender_labels = ['Man', 'Woman']
        race_labels = ['white', 'black', 'asian', 'indian', 'others']
        
        return {
            'age': age,
            'gender': gender_labels[gender_id],
            'dominant_emotion': 'neutral',
            'emotion': {
                'neutral': 1.0
            },
            'dominant_race': race_labels[race_id],
            'race': {
                race_labels[race_id]: 1.0
            }
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Fallback
        return {
            'age': 30,
            'gender': 'Unknown',
            'dominant_emotion': 'neutral',
            'dominant_race': 'unknown',
            'error': str(e)
        }

