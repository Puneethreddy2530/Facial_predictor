
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add current directory to path so we can import if needed, but we'll define classes here for safety
sys.path.append(os.getcwd())

# Copying classes from train_pytorch.py to ensure compatibility and standalone execution
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.file_paths = []
        self.labels = []
        valid_extensions = ['*.jpg', '*.png', '*.jpeg']
        files = []
        for ext in valid_extensions:
            files.extend(list(self.root_dir.glob(f"**/{ext}")))
            
        for file_path in files:
            try:
                parts = file_path.stem.split('_')
                if len(parts) < 3: continue
                age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
                if age < 0 or age > 116 or gender not in [0, 1] or race not in range(5): continue
                self.file_paths.append(str(file_path))
                self.labels.append({'age': float(age), 'gender': float(gender), 'race': int(race)})
            except: continue

    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image)
        l = self.labels[idx]
        return image, {'age': torch.tensor(l['age'], dtype=torch.float32), 
                       'gender': torch.tensor(l['gender'], dtype=torch.float32), 
                       'race': torch.tensor(l['race'], dtype=torch.long)}

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, pretrained=False):
        super(MultiTaskEfficientNet, self).__init__()
        backbone = models.efficientnet_b2(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        fc_in_features = 1408 
        self.age_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fc_in_features, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 1))
        self.gender_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fc_in_features, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 1))
        self.race_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(fc_in_features, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 5))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return {'age': self.age_head(x), 'gender': self.gender_head(x), 'race': self.race_head(x)}

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset path resolution
    dataset_path = Path("datasets/UTKFace")
    if not dataset_path.exists(): dataset_path = Path("datasets/utkface_aligned_cropped/UTKFace")
    if not dataset_path.exists():
        print("Dataset not found!")
        return

    full_dataset = UTKFaceDataset(dataset_path, transform=val_transform)
    # Replicate split logic roughly or just evaluate on a subset
    # We'll use a random subset for "validation" if we can't replicate exact split
    # Logic in training was: 85% train, rest val.
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = MultiTaskEfficientNet().to(device)
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
        print("Loaded models/best_model.pth")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    
    criterion_age = nn.SmoothL1Loss()
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_race = nn.CrossEntropyLoss()
    
    total_samples = 0
    total_loss = 0.0
    age_mae_sum = 0.0
    gender_acc_sum = 0
    race_acc_sum = 0
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            age_labels = labels['age'].to(device).unsqueeze(1)
            gender_labels = labels['gender'].to(device).unsqueeze(1)
            race_labels = labels['race'].to(device)
            
            outputs = model(images)
            
            # Loss
            l_age = criterion_age(outputs['age'], age_labels)
            l_gender = criterion_gender(outputs['gender'], gender_labels)
            l_race = criterion_race(outputs['race'], race_labels)
            total_loss += (l_age + l_gender + l_race).item() * images.size(0)
            
            # MAE
            age_mae_sum += torch.abs(outputs['age'] - age_labels).sum().item()
            
            # Accuracy
            gender_preds = (torch.sigmoid(outputs['gender']) > 0.5).float()
            gender_acc_sum += (gender_preds == gender_labels).sum().item()
            
            _, race_preds = torch.max(outputs['race'], 1)
            race_acc_sum += (race_preds == race_labels).sum().item()
            
            total_samples += images.size(0)
            
    if total_samples == 0:
        print("No samples evaluated.")
        return

    print(f"\nResults on {total_samples} validation samples:")
    print(f"Val Loss: {total_loss / total_samples:.4f}")
    print(f"Age MAE: {age_mae_sum / total_samples:.2f} years")
    # print(f"Gender Accuracy: {gender_acc_sum / total_samples * 100:.2f}%")
    # print(f"Race Accuracy: {race_acc_sum / total_samples * 100:.2f}%")
    
    # Store results in a file for the agent to read if stdout is captured poorly
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Val Loss: {total_loss / total_samples:.4f}\n")
        f.write(f"Age MAE: {age_mae_sum / total_samples:.2f}\n")
        f.write(f"Gender Accuracy: {gender_acc_sum / total_samples:.4f}\n")
        f.write(f"Race Accuracy: {race_acc_sum / total_samples:.4f}\n")
        
if __name__ == "__main__":
    evaluate()
