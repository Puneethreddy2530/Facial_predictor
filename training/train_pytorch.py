
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import glob
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
import copy

# ==========================================
# 1. Dataset Class
# ==========================================
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # UTKFace format: age_gender_race_date.jpg
        self.file_paths = []
        self.labels = []
        
        valid_extensions = ['*.jpg', '*.png', '*.jpeg']
        files = []
        for ext in valid_extensions:
            files.extend(list(self.root_dir.glob(f"**/{ext}")))
            
        print(f"Scanning {self.root_dir}...")
        
        for file_path in files:
            try:
                # Parse filename
                parts = file_path.stem.split('_')
                if len(parts) < 3:
                    continue
                
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                # Filter invalid data
                if age < 0 or age > 116 or gender not in [0, 1] or race not in range(5):
                    continue
                    
                self.file_paths.append(str(file_path))
                self.labels.append({
                    'age': float(age),
                    'gender': float(gender),
                    'race': int(race)
                })
                
            except Exception:
                continue
                
        print(f"Found {len(self.file_paths)} valid images.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        labels = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, {
            'age': torch.tensor(labels['age'], dtype=torch.float32),
            'gender': torch.tensor(labels['gender'], dtype=torch.float32),
            'race': torch.tensor(labels['race'], dtype=torch.long)
        }

# ==========================================
# 2. Multi-Task Model (The "Goated" Architecture)
# ==========================================
class MultiTaskEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MultiTaskEfficientNet, self).__init__()
        
        # Backbone: EfficientNet-B2 (Smart, Efficient, Accurate)
        print("Using EfficientNet-B2 Backbone (Optimum Speed/Acc)")
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b2(weights=weights)
        
        # EfficientNet features end with a pooling layer usually, but let's take features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        
        # Hidden dimension for B2 is 1408
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

# ==========================================
# 3. Training Function
# ==========================================
def train_model(data_dir, num_epochs=50, batch_size=32):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True 
        # RTX 30 Series "Physics" Optimization (TensorFloat-32)
        torch.set_float32_matmul_precision('high') 
        print("✓ Activated TensorFloat-32 (TF32) acceleration")
    
    # Transforms (Lightweight Augmentation for Speed)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)), # Downsize slightly for speed? No, keep 200
        transforms.RandomCrop(200, padding=20), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    full_dataset = UTKFaceDataset(data_dir, transform=train_transform)
    
    # Split
    train_size = int(0.85 * len(full_dataset)) # More training data
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = val_transform
    
    # DataLoaders - Optimized
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Model
    model = MultiTaskEfficientNet(pretrained=True).to(device)
    
    # PyTorch 2.0 Compilation (The "New Tech")
    # Note: torch.compile/Triton has issues on Windows, so we skip it to be safe
    if os.name != 'nt': 
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
            print("✓ Model compiled for extra speed")
        except Exception as e:
            print(f"Compilation skipped: {e}")
    else:
        print("Skipping torch.compile() (Windows optimization fallback)")
    
    # Losses
    criterion_age = nn.SmoothL1Loss() # Better convergence than L1
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_race = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    patience = 7 # Early stopping patience
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            age_labels = labels['age'].to(device, non_blocking=True).unsqueeze(1)
            gender_labels = labels['gender'].to(device, non_blocking=True).unsqueeze(1)
            race_labels = labels['race'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True) # Slightly faster
            
            # Forward pass
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    l_age = criterion_age(outputs['age'], age_labels)
                    l_gender = criterion_gender(outputs['gender'], gender_labels)
                    l_race = criterion_race(outputs['race'], race_labels)
                    loss = l_age + l_gender + l_race
            else:
                 outputs = model(images)
                 l_age = criterion_age(outputs['age'], age_labels)
                 l_gender = criterion_gender(outputs['gender'], gender_labels)
                 l_race = criterion_race(outputs['race'], race_labels)
                 loss = l_age + l_gender + l_race    
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                age_labels = labels['age'].to(device).unsqueeze(1)
                gender_labels = labels['gender'].to(device).unsqueeze(1)
                race_labels = labels['race'].to(device)
                
                outputs = model(images)
                
                l_age = criterion_age(outputs['age'], age_labels)
                l_gender = criterion_gender(outputs['gender'], gender_labels)
                l_race = criterion_race(outputs['race'], race_labels)
                
                val_loss += (l_age + l_gender + l_race).item()
        
        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Early Stopping & Save
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = Path("models")
            save_path.mkdir(exist_ok=True)
            # Save the underlying model if compiled
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), save_path / "best_model.pth")
            print("★ New best model saved!")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\n✋ Early stopping triggered at Epoch {epoch+1}!")
            break
            
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    # Check for UTKFace dataset
    dataset_path = Path("datasets/UTKFace")
    
    # Try alternate paths if not found directly
    if not dataset_path.exists():
         # Maybe it's in a subdirectory
         dataset_path = Path("datasets/utkface_aligned_cropped/UTKFace")
         
    if not dataset_path.exists():
        print("Dataset not found. Please ensure UTKFace is in datasets/UTKFace")
    else:
        train_model(dataset_path, num_epochs=20)
