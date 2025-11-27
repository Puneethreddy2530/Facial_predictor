"""
Automated UTKFace dataset downloader using opendatasets library.
This bypasses the need for manual Kaggle API setup.
"""
import os
import sys

def download_with_opendatasets():
    """Download using opendatasets (interactive Kaggle login)"""
    try:
        import opendatasets as od
    except ImportError:
        print("Installing opendatasets...")
        os.system("pip install opendatasets")
        import opendatasets as od
    
    print("\n" + "="*60)
    print("  UTKFace Dataset Download")
    print("="*60)
    print("\nYou'll be prompted to enter your Kaggle credentials.")
    print("Get them from: https://www.kaggle.com/settings/account")
    print("(Click 'Create New Token' to download kaggle.json)")
    print("\n" + "="*60 + "\n")
    
    dataset_url = 'https://www.kaggle.com/datasets/jangedoo/utkface-new'
    
    try:
        od.download(dataset_url, data_dir='datasets')
        print("\n✓ Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n⚠ Download failed: {e}")
        return False

if __name__ == '__main__':
    success = download_with_opendatasets()
    
    if success:
        print("\n📊 Checking downloaded data...")
        dataset_path = 'datasets/utkface-new'
        if os.path.exists(dataset_path):
            files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]
            print(f"✓ Found {len(files)} images")
            print(f"Location: {dataset_path}")
            print("\nNext step: Run training")
            print("Command: python training\\train_sklearn_model.py")
        else:
            print("⚠ Dataset folder not found")
    else:
        print("\n📋 Manual download option:")
        print("1. Visit: https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print("2. Click Download (requires Kaggle login)")
        print("3. Extract to: datasets/utkface-new/")
