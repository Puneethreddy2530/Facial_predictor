"""
Model Training Pipeline
Train custom facial prediction models (age, gender, emotion, race)
"""
import os
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import json


class FacialPredictionTrainer:
    def __init__(self, dataset_path, model_save_dir="models"):
        self.dataset_path = Path(dataset_path)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        self.img_size = (128, 128)
        self.batch_size = 32
        
    def load_utkface_data(self):
        """
        Load UTKFace dataset
        Format: [age]_[gender]_[race]_[date&time].jpg
        gender: 0=male, 1=female
        race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
        """
        print("📂 Loading UTKFace dataset...")
        
        images = []
        ages = []
        genders = []
        races = []
        
        image_files = list(self.dataset_path.glob("**/*.jpg")) + \
                     list(self.dataset_path.glob("**/*.png"))
        
        for img_path in image_files:
            try:
                # Parse filename: age_gender_race_date.jpg
                parts = img_path.stem.split('_')
                if len(parts) < 3:
                    continue
                
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                # Skip invalid data
                if age < 0 or age > 116 or gender not in [0, 1] or race not in range(5):
                    continue
                
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                
                images.append(img)
                ages.append(age)
                genders.append(gender)
                races.append(race)
                
            except (ValueError, IndexError):
                continue
        
        print(f"✓ Loaded {len(images)} images")
        
        return {
            'images': np.array(images),
            'ages': np.array(ages),
            'genders': np.array(genders),
            'races': np.array(races)
        }
    
    def create_age_model(self, input_shape=(128, 128, 3)):
        """Create CNN model for age prediction (regression)"""
        model = keras.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='linear')  # Age output (0-116)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_gender_model(self, input_shape=(128, 128, 3)):
        """Create CNN model for gender prediction (binary classification)"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Gender: 0=male, 1=female
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_race_model(self, input_shape=(128, 128, 3), num_classes=5):
        """Create CNN model for race prediction (multi-class)"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')  # 5 race categories
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, data, epochs=50, validation_split=0.2):
        """Train all models"""
        print("\n🚀 Starting training...")
        
        X = data['images']
        X_train, X_val = train_test_split(X, test_size=validation_split, random_state=42)
        
        results = {}
        
        # Train Age Model
        print("\n📊 Training Age Model...")
        y_age = data['ages']
        y_age_train, y_age_val = train_test_split(y_age, test_size=validation_split, random_state=42)
        
        age_model = self.create_age_model()
        history_age = age_model.fit(
            X_train, y_age_train,
            validation_data=(X_val, y_age_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        age_model.save(self.model_save_dir / 'age_model.h5')
        results['age'] = history_age.history
        print("✓ Age model saved")
        
        # Train Gender Model
        print("\n⚥ Training Gender Model...")
        y_gender = data['genders']
        y_gender_train, y_gender_val = train_test_split(y_gender, test_size=validation_split, random_state=42)
        
        gender_model = self.create_gender_model()
        history_gender = gender_model.fit(
            X_train, y_gender_train,
            validation_data=(X_val, y_gender_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        gender_model.save(self.model_save_dir / 'gender_model.h5')
        results['gender'] = history_gender.history
        print("✓ Gender model saved")
        
        # Train Race Model
        print("\n🌍 Training Race Model...")
        y_race = data['races']
        y_race_train, y_race_val = train_test_split(y_race, test_size=validation_split, random_state=42)
        
        race_model = self.create_race_model()
        history_race = race_model.fit(
            X_train, y_race_train,
            validation_data=(X_val, y_race_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
        race_model.save(self.model_save_dir / 'race_model.h5')
        results['race'] = history_race.history
        print("✓ Race model saved")
        
        # Save training history
        with open(self.model_save_dir / 'training_history.json', 'w') as f:
            json.dump({k: {kk: [float(vv) for vv in v] for kk, v in vv.items()} 
                      for k, vv in results.items()}, f)
        
        print("\n✅ All models trained and saved!")
        print(f"Models location: {self.model_save_dir.absolute()}")
        
        return results


def main():
    print("=" * 60)
    print("  Facial Prediction Model Training")
    print("=" * 60)
    print()
    
    # Check for dataset
    dataset_path = Path("datasets/utkface")
    if not dataset_path.exists():
        print("⚠ Dataset not found!")
        print("Run 'python data_collection/scrape_faces.py' first")
        return
    
    # Initialize trainer
    trainer = FacialPredictionTrainer(dataset_path)
    
    # Load data
    data = trainer.load_utkface_data()
    
    if len(data['images']) == 0:
        print("⚠ No images found in dataset")
        return
    
    # Train models
    print(f"\n📊 Dataset size: {len(data['images'])} images")
    print("Starting training (this may take a while)...")
    
    results = trainer.train_models(data, epochs=30)
    
    print("\n🎉 Training complete!")
    print("Models saved in: models/")


if __name__ == "__main__":
    main()
