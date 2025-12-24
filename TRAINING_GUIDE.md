# Training & Data Collection Setup Guide

## Complete Workflow: Scrape → Train → Deploy

### Step 1: Install Training Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# For Kaggle datasets, also setup:
pip install kaggle
```

**Setup Kaggle API** (for FER2013 emotion dataset):
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/` (or `%USERPROFILE%\.kaggle\` on Windows)

---

### Step 2: Collect Training Data

Run the data scraper:

```powershell
python data_collection/scrape_faces.py
```

**Available datasets:**
- **Option 1**: Sample (LFW) - 170MB, quick start
- **Option 2**: UTKFace - 23K images with age, gender, race (~500MB)
- **Option 3**: FER2013 - 35K images with emotions (~300MB, needs Kaggle)
- **Option 4**: FairFace - 100K balanced faces with age/gender/race (~5GB, Kaggle)

**Manual downloads** (if automated fails):
- UTKFace: https://susanqq.github.io/UTKFace/
- FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
- FairFace: https://www.kaggle.com/datasets/djagatiya/fairface
- LFW: http://vis-www.cs.umass.edu/lfw/lfw.tgz

---

### Step 3: Train Custom Models

**Fast upgrade (recommended): Multitask EfficientNet**

```powershell
# Train on UTKFace (age/gender/race) + FER2013 (emotion)
python training/train_multitask.py ^
    --utkface-dir datasets/utkface_aligned_cropped/UTKFace ^
    --fer-csv datasets/fer2013.csv ^
    --epochs 20 --batch-size 32
```

Produces:
- `models/multitask_model.keras` – shared backbone with 4 heads
- `models/multitask_meta.json` – labels + image size
- `models/multitask_history.json` – training curves

Notes:
- Uses EfficientNetB0 with data augmentation and early stopping
- Needs both datasets; set `--limit-utk`/`--limit-fer` for quick smoke tests
- Backend auto-loads this model when present (`USE_CUSTOM_MODEL=1`, default)

**Legacy CNN (single-task per head)**

```powershell
python training/train_model.py
```

This will:
- Load dataset from `datasets/`
- Train 3 models: Age, Gender, Race
- Save models to `models/` directory
- Take 30-60 minutes on CPU (faster with GPU)

**Training output:**
- `models/age_model.h5` - Age prediction
- `models/gender_model.h5` - Gender classification
- `models/race_model.h5` - Race/ethnicity classification
- `models/training_history.json` - Training metrics

---

### Step 4: Use Custom Models in Backend

Update `backend/core.py` to load your trained models instead of DeepFace:

```python
# Add to backend/core.py
import tensorflow as tf

# Load custom models
age_model = tf.keras.models.load_model('models/age_model.h5')
gender_model = tf.keras.models.load_model('models/gender_model.h5')
race_model = tf.keras.models.load_model('models/race_model.h5')

def predict_with_custom_models(image):
    # Preprocess
    img = cv2.resize(image, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    age = age_model.predict(img)[0][0]
    gender_prob = gender_model.predict(img)[0][0]
    race_probs = race_model.predict(img)[0]
    
    return {
        'age': int(age),
        'gender': 'Woman' if gender_prob > 0.5 else 'Man',
        'race': ['White', 'Black', 'Asian', 'Indian', 'Others'][race_probs.argmax()]
    }
```

---

### Step 5: View Training Results

```powershell
python training/visualize_results.py
```

Shows:
- Training/validation curves
- Model accuracy metrics
- Sample predictions

---

## Quick Start (Testing)

For immediate testing without training:

```powershell
# Use mock mode (current setup)
$env:MOCK_PREDICTION='1'
.\start_backend.ps1
```

Or download pre-trained models:
```powershell
# Create models directory
mkdir models

# Download pre-trained weights (example)
# Add your model download links here
```

---

## Dataset Information

### UTKFace Format
Filename: `[age]_[gender]_[race]_[datetime].jpg`
- Age: 0-116
- Gender: 0=male, 1=female
- Race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others

### Training Tips
- Start with 10K images minimum for decent accuracy
- Use GPU if available (50x faster)
- Monitor validation loss to avoid overfitting
- Augment data (flip, rotate, brightness) for better generalization

---

## Troubleshooting

**Out of memory:**
- Reduce `batch_size` in train_model.py
- Use smaller image size (64x64 instead of 128x128)

**Low accuracy:**
- Train for more epochs
- Collect more diverse data
- Add data augmentation
- Try transfer learning (ResNet, MobileNet)

**Download fails:**
- Check internet connection
- Try manual download links
- Use VPN if region-blocked
