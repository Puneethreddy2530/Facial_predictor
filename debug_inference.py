
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
import sys
import os

# Add path to import backend modules
sys.path.append(os.getcwd())

try:
    from backend.inference_pytorch import FacialPredictor
except ImportError:
    print("Could not import FacialPredictor")
    sys.exit(1)

def debug_image(image_path):
    print(f"Debugging image: {image_path}")
    
    # Initialize predictor
    predictor = FacialPredictor(model_path='models/best_model.pth')
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load image: {e}")
        return

    # 1. Standard Prediction (Current Logic)
    print("\n--- Standard Prediction ---")
    results = predictor.predict(img)
    if not results:
        print("No faces detected.")
        return

    for res in results:
        print(f"Age: {res['age']}")
        print(f"Gender: {res['gender']} (Conf: {res['gender_conf']:.4f})")
        print(f"Race: {res['race']}")
        print(f"Box: {res['box']}")


if __name__ == "__main__":
    # Hardcoded path to the new uploaded image
    img_path = "C:/Users/punee/.gemini/antigravity/brain/9bfd4e80-e3d2-4d35-8b55-d5a571e2a8e3/uploaded_image_1766588540833.png"
    debug_image(img_path)
