"""
Visualize training results and test the trained models.
"""
import os
import json
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

class ModelVisualizer:
    def __init__(self, models_dir='models', data_dir='datasets/utkface_synthetic'):
        self.models_dir = models_dir
        self.data_dir = data_dir
        
    def plot_training_results(self):
        """Plot training metrics"""
        print("📊 Visualizing Training Results...\n")
        
        # Load results
        results_file = os.path.join(self.models_dir, 'training_results.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Facial Prediction Model Performance', fontsize=16, fontweight='bold')
        
        # Age Model
        axes[0].bar(['Age MAE'], [results['age_mae']], color='#3498db')
        axes[0].set_ylabel('Mean Absolute Error (years)')
        axes[0].set_title('Age Prediction')
        axes[0].set_ylim(0, max(20, results['age_mae'] + 5))
        axes[0].text(0, results['age_mae'] + 0.5, f"{results['age_mae']:.2f}", 
                    ha='center', va='bottom', fontweight='bold')
        
        # Gender Model
        axes[1].bar(['Gender Accuracy'], [results['gender_accuracy'] * 100], color='#2ecc71')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Gender Prediction')
        axes[1].set_ylim(0, 100)
        axes[1].text(0, results['gender_accuracy'] * 100 + 1, f"{results['gender_accuracy']*100:.1f}%", 
                    ha='center', va='bottom', fontweight='bold')
        
        # Race Model
        axes[2].bar(['Race Accuracy'], [results['race_accuracy'] * 100], color='#e74c3c')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('Race Prediction')
        axes[2].set_ylim(0, 100)
        axes[2].text(0, results['race_accuracy'] * 100 + 1, f"{results['race_accuracy']*100:.1f}%", 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(self.models_dir, 'training_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {output_path}")
        plt.close()
        
        # Print summary
        print("\n" + "="*60)
        print("  TRAINING SUMMARY")
        print("="*60)
        print(f"  Total Samples: {results['num_samples']}")
        print(f"  Training Set: {results['train_size']}")
        print(f"  Test Set: {results['test_size']}")
        print(f"  Trained: {results['timestamp']}")
        print("\n  Model Performance:")
        print(f"    Age MAE: {results['age_mae']:.2f} years")
        print(f"    Gender Accuracy: {results['gender_accuracy']*100:.2f}%")
        print(f"    Race Accuracy: {results['race_accuracy']*100:.2f}%")
        print("="*60 + "\n")
    
    def test_models_on_samples(self, num_samples=5):
        """Test trained models on sample images"""
        print("🧪 Testing Models on Sample Images...\n")
        
        # Load models
        age_model = joblib.load(os.path.join(self.models_dir, 'age_model.pkl'))
        gender_model = joblib.load(os.path.join(self.models_dir, 'gender_model.pkl'))
        race_model = joblib.load(os.path.join(self.models_dir, 'race_model.pkl'))
        
        # Get sample images
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')][:num_samples]
        
        # Labels
        gender_labels = ['Male', 'Female']
        race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for idx, filename in enumerate(files):
            # Parse ground truth
            parts = filename.split('_')
            true_age = int(parts[0])
            true_gender = int(parts[1])
            true_race = int(parts[2])
            
            # Load and preprocess image
            img_path = os.path.join(self.data_dir, filename)
            img = cv2.imread(img_path)
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Prepare for prediction
            img_resized = cv2.resize(img, (64, 64))
            img_flat = img_resized.flatten() / 255.0
            
            # Predict
            pred_age = int(age_model.predict([img_flat])[0])
            pred_gender = int(gender_model.predict([img_flat])[0])
            pred_race = int(race_model.predict([img_flat])[0])
            
            # Display
            axes[idx].imshow(img_display)
            axes[idx].axis('off')
            
            # Title with predictions
            title = f"True: {true_age}y, {gender_labels[true_gender]}, {race_labels[true_race]}\n"
            title += f"Pred: {pred_age}y, {gender_labels[pred_gender]}, {race_labels[pred_race]}"
            
            # Color: green if all correct, yellow if partial, red if all wrong
            correct = sum([pred_age == true_age, pred_gender == true_gender, pred_race == true_race])
            color = '#2ecc71' if correct == 3 else '#f39c12' if correct >= 1 else '#e74c3c'
            
            axes[idx].set_title(title, fontsize=10, color=color, fontweight='bold')
            
            print(f"Image {idx+1}: {filename}")
            print(f"  True: Age={true_age}, Gender={gender_labels[true_gender]}, Race={race_labels[true_race]}")
            print(f"  Pred: Age={pred_age}, Gender={gender_labels[pred_gender]}, Race={race_labels[pred_race]}")
            print()
        
        plt.tight_layout()
        output_path = os.path.join(self.models_dir, 'sample_predictions.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved predictions: {output_path}\n")
        plt.close()

if __name__ == '__main__':
    visualizer = ModelVisualizer()
    visualizer.plot_training_results()
    visualizer.test_models_on_samples(num_samples=5)
