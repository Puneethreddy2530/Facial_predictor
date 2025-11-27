"""
Face Image Scraper
Collects face images from public datasets and sources for training.
Uses ethical sources: UTKFace, IMDB-WIKI, CelebA (with proper attribution)
"""
import os
import requests
from pathlib import Path
import zipfile
import gdown
from tqdm import tqdm


class FaceDataScraper:
    def __init__(self, data_dir="datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_utkface(self):
        """
        Download UTKFace dataset (23,708 face images with age, gender, ethnicity)
        Source: https://susanqq.github.io/UTKFace/
        """
        print("📥 Downloading UTKFace dataset...")
        
        # UTKFace dataset URLs (parts 1, 2, 3)
        urls = [
            "https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk",  # Part 1
            "https://drive.google.com/uc?id=0BxYys69jI14kS0ZxcmlxRFBJSlE",  # Part 2
            "https://drive.google.com/uc?id=0BxYys69jI14kMTlfNkNzSHdIUGc",  # Part 3
        ]
        
        output_dir = self.data_dir / "utkface"
        output_dir.mkdir(exist_ok=True)
        
        for i, url in enumerate(urls, 1):
            output_file = output_dir / f"utkface_part{i}.tar.gz"
            if output_file.exists():
                print(f"✓ Part {i} already exists")
                continue
                
            try:
                print(f"Downloading part {i}/3...")
                gdown.download(url, str(output_file), quiet=False)
                print(f"✓ Part {i} downloaded")
            except Exception as e:
                print(f"⚠ Error downloading part {i}: {e}")
                print("Note: You may need to manually download from https://susanqq.github.io/UTKFace/")
        
        return output_dir
    
    def download_fer2013(self):
        """
        Download FER2013 emotion dataset (35,887 grayscale face images)
        Source: Kaggle - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
        """
        print("📥 Downloading FER2013 emotion dataset...")
        print("⚠ FER2013 requires Kaggle API credentials")
        print("Setup: pip install kaggle")
        print("Get API key from: https://www.kaggle.com/settings")
        
        output_dir = self.data_dir / "fer2013"
        output_dir.mkdir(exist_ok=True)
        
        try:
            import kaggle
            print("Downloading from Kaggle...")
            kaggle.api.competition_download_files(
                'challenges-in-representation-learning-facial-expression-recognition-challenge',
                path=str(output_dir),
                quiet=False
            )
            print("✓ FER2013 downloaded")
        except ImportError:
            print("⚠ Kaggle package not installed. Install with: pip install kaggle")
        except Exception as e:
            print(f"⚠ Error: {e}")
            print("Manual download: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
        
        return output_dir
    
    def download_sample_dataset(self):
        """
        Download a small sample dataset for quick testing
        """
        print("📥 Downloading sample face dataset...")
        
        # Using a smaller, publicly available dataset
        url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"  # Labeled Faces in the Wild
        output_dir = self.data_dir / "lfw"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "lfw.tgz"
        
        if output_file.exists():
            print("✓ Sample dataset already exists")
            return output_dir
        
        try:
            print("Downloading Labeled Faces in the Wild...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc="LFW"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            # Extract
            print("Extracting...")
            import tarfile
            with tarfile.open(output_file, 'r:gz') as tar:
                tar.extractall(output_dir)
            
            print("✓ Sample dataset ready")
        except Exception as e:
            print(f"⚠ Error: {e}")
        
        return output_dir
    
    def get_dataset_info(self):
        """Display information about downloaded datasets"""
        print("\n📊 Dataset Information:")
        print("=" * 60)
        
        datasets = {
            "utkface": "UTKFace - 23K+ images with age, gender, race",
            "fer2013": "FER2013 - 35K+ images with emotions",
            "lfw": "Labeled Faces in the Wild - 13K+ images"
        }
        
        for name, desc in datasets.items():
            path = self.data_dir / name
            if path.exists():
                count = len(list(path.rglob("*.jpg"))) + len(list(path.rglob("*.png")))
                print(f"✓ {name}: {count} images")
                print(f"  {desc}")
            else:
                print(f"✗ {name}: Not downloaded")
                print(f"  {desc}")
        print("=" * 60)


def main():
    print("=" * 60)
    print("  Face Dataset Scraper")
    print("=" * 60)
    print()
    
    scraper = FaceDataScraper()
    
    print("Select dataset to download:")
    print("1. Sample dataset (LFW) - ~170MB - Quick start")
    print("2. UTKFace - Age, Gender, Race - ~500MB")
    print("3. FER2013 - Emotions - ~300MB (requires Kaggle)")
    print("4. All datasets")
    print("5. Show dataset info")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        scraper.download_sample_dataset()
    elif choice == "2":
        scraper.download_utkface()
    elif choice == "3":
        scraper.download_fer2013()
    elif choice == "4":
        scraper.download_sample_dataset()
        scraper.download_utkface()
        scraper.download_fer2013()
    elif choice == "5":
        scraper.get_dataset_info()
    else:
        print("Invalid choice")
    
    print("\n✓ Complete!")
    print(f"Data saved to: {scraper.data_dir.absolute()}")


if __name__ == "__main__":
    main()
