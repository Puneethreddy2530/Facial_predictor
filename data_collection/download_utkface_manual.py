"""
Download UTKFace dataset directly without Kaggle API.
Uses alternative sources for the dataset.
"""
import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

def download_utkface():
    """Download UTKFace from alternative source"""
    print("🔍 Downloading UTKFace Dataset...")
    print("Source: Alternative mirror")
    print("")
    
    os.makedirs('datasets', exist_ok=True)
    
    # Try multiple sources
    sources = [
        {
            'name': 'UTKFace Part 1',
            'url': 'https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk',
            'file': 'datasets/utkface_part1.tar.gz'
        },
        {
            'name': 'UTKFace Part 2', 
            'url': 'https://drive.google.com/uc?id=0BxYys69jI14kS0ZxcmlxRFBJSlE',
            'file': 'datasets/utkface_part2.tar.gz'
        },
        {
            'name': 'UTKFace Part 3',
            'url': 'https://drive.google.com/uc?id=0BxYys69jI14kMTlfNkNzSHdIUGc',
            'file': 'datasets/utkface_part3.tar.gz'
        }
    ]
    
    print("⚠ Google Drive links require manual download.")
    print("")
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 60)
    print("")
    print("Option 1: Direct Download (RECOMMENDED)")
    print("-" * 60)
    print("1. Go to: https://www.kaggle.com/datasets/jangedoo/utkface-new")
    print("2. Click 'Download' button (requires Kaggle account)")
    print("3. Save to: C:\\Users\\punee\\Downloads\\")
    print("4. Extract to: C:\\Users\\punee\\OneDrive\\Desktop\\Facial_prediction\\datasets\\")
    print("")
    print("Option 2: Use Kaggle CLI")
    print("-" * 60)
    print("Run: .\\setup_kaggle.ps1")
    print("(Follow instructions to set up Kaggle API)")
    print("")
    print("Option 3: Alternative Dataset")
    print("-" * 60)
    print("Visit: https://susanqq.github.io/UTKFace/")
    print("Download aligned & cropped faces")
    print("")
    print("=" * 60)
    print("")
    
    input("Press Enter after downloading and extracting the dataset...")
    
    # Check if dataset exists
    possible_paths = [
        'datasets/utkface-new',
        'datasets/UTKFace',
        'datasets/utkface',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith('.jpg')]
            if len(files) > 0:
                print(f"✓ Found {len(files)} images in {path}")
                return path
    
    print("⚠ Dataset not found. Please extract to datasets/ folder")
    return None

if __name__ == '__main__':
    download_utkface()
