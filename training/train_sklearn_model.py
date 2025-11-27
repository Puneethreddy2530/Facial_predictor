"""
Train facial prediction models using scikit-learn (Python 3.14 compatible).
Trains Random Forest models for age, gender, and race prediction.
"""
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import joblib
import json
from datetime import datetime

class FacialPredictionTrainer:
    def __init__(self, data_dir='datasets/UTKFace', output_dir='models'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess images from UTKFace format"""
        print("📂 Loading images from", self.data_dir)
        
        images = []
        ages = []
        genders = []
        races = []
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        print(f"Found {len(files)} images")
        
        for i, filename in enumerate(files):
            try:
                # Parse filename: [age]_[gender]_[race]_[timestamp].jpg
                parts = filename.split('_')
                if len(parts) < 3:
                    continue
                
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                # Load and preprocess image
                img_path = os.path.join(self.data_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Convert to grayscale for better feature extraction
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize to 64x64 and flatten
                img_resized = cv2.resize(gray, (64, 64))
                img_flat = img_resized.flatten() / 255.0  # Normalize
                
                images.append(img_flat)
                ages.append(age)
                genders.append(gender)
                races.append(race)
                
                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1}/{len(files)} images")
                    
            except Exception as e:
                print(f"  ⚠ Error loading {filename}: {e}")
                continue
        
        print(f"\n✓ Successfully loaded {len(images)} images")
        return np.array(images), np.array(ages), np.array(genders), np.array(races)
    
    def train_models(self):
        """Train Random Forest models for age, gender, and race"""
        print("\n" + "="*60)
        print("  TRAINING FACIAL PREDICTION MODELS")
        print("="*60 + "\n")
        
        # Load data
        X, y_age, y_gender, y_race = self.load_data()
        
        # Split data
        print("\n📊 Splitting data into train/test sets (80/20)...")
        X_train, X_test, age_train, age_test = train_test_split(
            X, y_age, test_size=0.2, random_state=42
        )
        _, _, gender_train, gender_test = train_test_split(
            X, y_gender, test_size=0.2, random_state=42
        )
        _, _, race_train, race_test = train_test_split(
            X, y_race, test_size=0.2, random_state=42
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        
        results = {}
        
        # Train Age Model (Regression)
        print("\n" + "-"*60)
        print("🔢 Training Age Prediction Model (Random Forest Regressor)")
        print("-"*60)
        age_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=30,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        age_model.fit(X_train, age_train)
        age_pred = age_model.predict(X_test)
        age_mae = mean_absolute_error(age_test, age_pred)
        print(f"\n✓ Age Model MAE: {age_mae:.2f} years")
        results['age_mae'] = float(age_mae)
        
        # Save model
        joblib.dump(age_model, os.path.join(self.output_dir, 'age_model.pkl'))
        print(f"  Saved: models/age_model.pkl")
        
        # Train Gender Model (Binary Classification)
        print("\n" + "-"*60)
        print("👤 Training Gender Prediction Model (Random Forest Classifier)")
        print("-"*60)
        gender_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        gender_model.fit(X_train, gender_train)
        gender_pred = gender_model.predict(X_test)
        gender_acc = accuracy_score(gender_test, gender_pred)
        print(f"\n✓ Gender Model Accuracy: {gender_acc*100:.2f}%")
        results['gender_accuracy'] = float(gender_acc)
        
        # Save model
        joblib.dump(gender_model, os.path.join(self.output_dir, 'gender_model.pkl'))
        print(f"  Saved: models/gender_model.pkl")
        
        # Train Race Model (Multi-class Classification)
        print("\n" + "-"*60)
        print("🌍 Training Race Prediction Model (Random Forest Classifier)")
        print("-"*60)
        race_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        race_model.fit(X_train, race_train)
        race_pred = race_model.predict(X_test)
        race_acc = accuracy_score(race_test, race_pred)
        print(f"\n✓ Race Model Accuracy: {race_acc*100:.2f}%")
        results['race_accuracy'] = float(race_acc)
        
        # Save model
        joblib.dump(race_model, os.path.join(self.output_dir, 'race_model.pkl'))
        print(f"  Saved: models/race_model.pkl")
        
        # Save training summary
        results['timestamp'] = datetime.now().isoformat()
        results['num_samples'] = len(X)
        results['train_size'] = len(X_train)
        results['test_size'] = len(X_test)
        
        with open(os.path.join(self.output_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETE!")
        print("="*60)
        print(f"\n📊 Final Results:")
        print(f"  Age MAE: {age_mae:.2f} years")
        print(f"  Gender Accuracy: {gender_acc*100:.2f}%")
        print(f"  Race Accuracy: {race_acc*100:.2f}%")
        print(f"\n💾 Models saved to: {self.output_dir}/")
        print(f"  - age_model.pkl")
        print(f"  - gender_model.pkl")
        print(f"  - race_model.pkl")
        print(f"  - training_results.json")
        
        return results

if __name__ == '__main__':
    trainer = FacialPredictionTrainer()
    trainer.train_models()
