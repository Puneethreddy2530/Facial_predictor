"""
Visualize Training Results
Shows training curves and sample predictions
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf


def plot_training_history(history_file="models/training_history.json"):
    """Plot training and validation metrics"""
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Training Results', fontsize=16)
    
    # Age Model
    if 'age' in history:
        ax = axes[0, 0]
        ax.plot(history['age']['loss'], label='Train Loss')
        ax.plot(history['age']['val_loss'], label='Val Loss')
        ax.set_title('Age Model - Loss (MSE)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        ax = axes[1, 0]
        ax.plot(history['age']['mae'], label='Train MAE')
        ax.plot(history['age']['val_mae'], label='Val MAE')
        ax.set_title('Age Model - MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Absolute Error (years)')
        ax.legend()
        ax.grid(True)
    
    # Gender Model
    if 'gender' in history:
        ax = axes[0, 1]
        ax.plot(history['gender']['loss'], label='Train Loss')
        ax.plot(history['gender']['val_loss'], label='Val Loss')
        ax.set_title('Gender Model - Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        ax = axes[1, 1]
        ax.plot(history['gender']['accuracy'], label='Train Acc')
        ax.plot(history['gender']['val_accuracy'], label='Val Acc')
        ax.set_title('Gender Model - Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)
    
    # Race Model
    if 'race' in history:
        ax = axes[0, 2]
        ax.plot(history['race']['loss'], label='Train Loss')
        ax.plot(history['race']['val_loss'], label='Val Loss')
        ax.set_title('Race Model - Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        ax = axes[1, 2]
        ax.plot(history['race']['accuracy'], label='Train Acc')
        ax.plot(history['race']['val_accuracy'], label='Val Acc')
        ax.set_title('Race Model - Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150)
    print("✓ Training curves saved to: models/training_curves.png")
    plt.show()


def test_models_on_samples(model_dir="models", dataset_dir="datasets/utkface", num_samples=6):
    """Test trained models on sample images"""
    
    # Load models
    age_model = tf.keras.models.load_model(f"{model_dir}/age_model.h5")
    gender_model = tf.keras.models.load_model(f"{model_dir}/gender_model.h5")
    race_model = tf.keras.models.load_model(f"{model_dir}/race_model.h5")
    
    race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']
    gender_labels = ['Male', 'Female']
    
    # Get sample images
    image_files = list(Path(dataset_dir).glob("**/*.jpg"))[:num_samples]
    
    if not image_files:
        print("No images found in dataset")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Predictions on Test Images', fontsize=16)
    
    for idx, img_path in enumerate(image_files[:6]):
        # Parse ground truth
        parts = img_path.stem.split('_')
        true_age = int(parts[0])
        true_gender = int(parts[1])
        true_race = int(parts[2])
        
        # Load and preprocess
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_norm = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)
        
        # Predict
        pred_age = int(age_model.predict(img_batch, verbose=0)[0][0])
        pred_gender_prob = gender_model.predict(img_batch, verbose=0)[0][0]
        pred_gender = 1 if pred_gender_prob > 0.5 else 0
        pred_race = int(race_model.predict(img_batch, verbose=0)[0].argmax())
        
        # Display
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img_rgb)
        ax.axis('off')
        
        title = f"True: {true_age}y, {gender_labels[true_gender]}, {race_labels[true_race]}\n"
        title += f"Pred: {pred_age}y, {gender_labels[pred_gender]}, {race_labels[pred_race]}"
        ax.set_title(title, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('models/sample_predictions.png', dpi=150)
    print("✓ Sample predictions saved to: models/sample_predictions.png")
    plt.show()


def print_model_summary():
    """Print model architectures"""
    print("=" * 60)
    print("  Model Summaries")
    print("=" * 60)
    
    models = {
        'Age': 'models/age_model.h5',
        'Gender': 'models/gender_model.h5',
        'Race': 'models/race_model.h5'
    }
    
    for name, path in models.items():
        if Path(path).exists():
            print(f"\n{name} Model:")
            model = tf.keras.models.load_model(path)
            print(f"  Parameters: {model.count_params():,}")
            print(f"  Layers: {len(model.layers)}")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
        else:
            print(f"\n{name} Model: Not found")


def main():
    print("=" * 60)
    print("  Training Results Visualization")
    print("=" * 60)
    print()
    
    if not Path("models/training_history.json").exists():
        print("⚠ No training history found!")
        print("Run 'python training/train_model.py' first")
        return
    
    print("1. Plot training curves")
    print("2. Test models on sample images")
    print("3. Show model summaries")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ["1", "4"]:
        plot_training_history()
    
    if choice in ["2", "4"]:
        test_models_on_samples()
    
    if choice in ["3", "4"]:
        print_model_summary()
    
    print("\n✓ Complete!")


if __name__ == "__main__":
    main()
