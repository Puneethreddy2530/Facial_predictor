
import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2
import os

# Re-defining the architecture here to be self-contained for deployment
class MultiTaskEfficientNet(nn.Module):
    def __init__(self, pretrained=False):
        super(MultiTaskEfficientNet, self).__init__()
        # Backbone EfficientNet-B2
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        
        fc_in_features = 1408
        
        # Heads
        self.age_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fc_in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
        self.gender_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fc_in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        
        self.race_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fc_in_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return {
            'age': self.age_head(x),
            'gender': self.gender_head(x),
            'race': self.race_head(x)
        }

class FacialPredictor:
    def __init__(self, model_path='models/best_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading FacialPredictor on {self.device}...")
        
        # Load Model
        self.model = MultiTaskEfficientNet(pretrained=False)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            # Handle compiled model keys
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[10:] if k.startswith('_orig_mod.') else k
                new_state_dict[name] = v
                
            try:
                self.model.load_state_dict(new_state_dict)
                print("✓ PyTorch model weights loaded.")
            except Exception as e:
                 print(f"⚠ Architecture mismatch (using random weights): {e}")
        else:
            print(f"⚠ Model not found at {model_path}. Using random weights for testing.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize Face Detector (MTCNN)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        print("✓ MTCNN Face Detector initialized.")
        
        # Transforms for the model
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']

    def predict(self, image):
        """
        Predict age, gender, race for all faces in the PIL image using TTA.
        """
        # 1. Detect faces
        try:
            boxes, probs = self.mtcnn.detect(image)
        except Exception as e:
            print(f"MTCNN Error: {e}")
            return []
            
        if boxes is None:
            return []
            
        results = []
        
        for i, box in enumerate(boxes):
            if probs[i] < 0.90: # Confidence threshold
                continue
                
            x1, y1, x2, y2 = [int(b) for b in box]

            # Add 30% margin for context
            w = x2 - x1
            h = y2 - y1
            margin_w = int(w * 0.0)
            margin_h = int(h * 0.0)
            
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(image.width, x2 + margin_w)
            y2 = min(image.height, y2 + margin_h)
            
            # Crop face
            face_img = image.crop((x1, y1, x2, y2))

            # Enhancement: CLAHE removed to match training data distribution
            if face_img.mode != 'RGB':
                face_img = face_img.convert('RGB')
            
            if face_img.size[0] < 20 or face_img.size[1] < 20:
                continue
            
            # TTA: Test Time Augmentation
            # 1. Original
            t1 = self.transform(face_img).unsqueeze(0).to(self.device)
            # 2. Horizontal Flip
            t2 = self.transform(face_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(self.device)
            
            # Inference (Average predictions)
            with torch.no_grad():
                out1 = self.model(t1)
                out2 = self.model(t2)
                
                # Average Age
                age = (out1['age'].item() + out2['age'].item()) / 2
                
                # Average Gender Logits
                gender_logit = (out1['gender'] + out2['gender']) / 2
                gender_score = torch.sigmoid(gender_logit).item()
                gender = "Woman" if gender_score > 0.5 else "Man"
                gender_conf = gender_score if gender == "Woman" else 1 - gender_score
                
                # Average Race Logits
                race_logits = (out1['race'] + out2['race']) / 2
                race_probs = torch.softmax(race_logits.squeeze(), dim=0)
                race_idx = torch.argmax(race_probs).item()
                race = self.race_labels[race_idx]
                
                # Format scores for JSON
                race_scores = {
                    self.race_labels[j].lower(): float(race_probs[j].item()) * 100 
                    for j in range(5)
                }
                
                # Mock emotions
                emotion_scores = {'neutral': 90.0, 'happy': 5.0, 'sad': 5.0} 
                
            results.append({
                'box': [x1, y1, x2-x1, y2-y1], # x, y, w, h
                'confidence': float(probs[i]),
                'age': int(round(age)),
                'gender': gender,
                'gender_conf': gender_conf,
                'race': race,
                'race_scores': race_scores,
                'emotion': 'Neutral', # Placeholder
                'emotion_scores': emotion_scores
            })
            
        return results
