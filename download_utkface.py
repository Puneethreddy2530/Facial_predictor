"""
Direct Kaggle dataset downloader with manual credential input.
"""
import os
import json

print("\n" + "="*70)
print("  UTKFace Dataset Download from Kaggle")
print("="*70)
print("\n📋 STEP 1: Get your Kaggle API credentials")
print("-"*70)
print("1. Go to: https://www.kaggle.com/settings/account")
print("2. Scroll to 'API' section")
print("3. Click 'Create New Token' (downloads kaggle.json)")
print("4. Open the file and copy the username and key")
print("")

username = input("Enter your Kaggle username: ").strip()
key = input("Enter your Kaggle API key: ").strip()

# Create kaggle config
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

kaggle_json = {
    "username": username,
    "key": key
}

config_path = os.path.join(kaggle_dir, "kaggle.json")
with open(config_path, 'w') as f:
    json.dump(kaggle_json, f)

print(f"\n✓ Kaggle config saved to: {config_path}")

# Set permissions (Windows)
print("\n📥 STEP 2: Downloading UTKFace dataset...")
print("-"*70)
print("Dataset: ~500MB, 23,708 face images")
print("This may take 5-10 minutes...")
print("")

os.chdir("C:\\Users\\punee\\OneDrive\\Desktop\\Facial_prediction")

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    print("✓ Authentication successful!")
    print("Downloading...")
    
    api.dataset_download_files(
        'jangedoo/utkface-new',
        path='datasets',
        unzip=True
    )
    
    print("\n✓ Dataset downloaded successfully!")
    
    # Check files
    dataset_path = 'datasets/utkface-new'
    if os.path.exists(dataset_path):
        files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        print(f"✓ Found {len(files)} images in {dataset_path}")
        print("\n" + "="*70)
        print("  READY TO TRAIN!")
        print("="*70)
        print("\nNext command:")
        print("  python training\\train_sklearn_model.py")
        print("")
    else:
        print(f"⚠ Expected path not found: {dataset_path}")
        print("Checking alternatives...")
        for root, dirs, files in os.walk('datasets'):
            jpg_files = [f for f in files if f.endswith('.jpg')]
            if len(jpg_files) > 1000:
                print(f"✓ Found {len(jpg_files)} images in {root}")
                break
        
except Exception as e:
    print(f"\n⚠ Error: {e}")
    print("\nAlternative option:")
    print("1. Visit: https://www.kaggle.com/datasets/jangedoo/utkface-new")
    print("2. Click 'Download' (298 MB)")
    print("3. Extract archive.zip to:")
    print("   C:\\Users\\punee\\OneDrive\\Desktop\\Facial_prediction\\datasets\\")
    print("4. Rename folder to 'utkface-new'")

print("\n" + "="*70)
