"""
Generate synthetic face images for training demonstration.
Creates realistic-looking test data when public datasets are unavailable.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

class SyntheticFaceGenerator:
    def __init__(self, output_dir='datasets/utkface_synthetic'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_face_image(self, age, gender, race):
        """Generate a synthetic face image with variation"""
        # Create 200x200 image
        img = Image.new('RGB', (200, 200), color=(230, 220, 210))
        draw = ImageDraw.Draw(img)
        
        # Base skin tone based on race
        skin_tones = {
            0: (255, 220, 177),  # White
            1: (0, 0, 0),        # Black
            2: (255, 205, 148),  # Asian
            3: (198, 134, 66),   # Indian
            4: (224, 172, 105),  # Others
        }
        base_color = skin_tones.get(race, (230, 220, 210))
        
        # Add noise for variation
        noise_r = random.randint(-20, 20)
        noise_g = random.randint(-20, 20)
        noise_b = random.randint(-20, 20)
        
        face_color = (
            max(0, min(255, base_color[0] + noise_r)),
            max(0, min(255, base_color[1] + noise_g)),
            max(0, min(255, base_color[2] + noise_b))
        )
        
        # Draw face oval
        draw.ellipse([30, 20, 170, 180], fill=face_color, outline=(0, 0, 0))
        
        # Draw eyes
        eye_y = 70 + random.randint(-5, 5)
        draw.ellipse([60, eye_y, 80, eye_y + 15], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.ellipse([120, eye_y, 140, eye_y + 15], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.ellipse([67, eye_y + 3, 73, eye_y + 12], fill=(50, 50, 50))
        draw.ellipse([127, eye_y + 3, 133, eye_y + 12], fill=(50, 50, 50))
        
        # Draw nose
        nose_y = 100 + random.randint(-5, 5)
        draw.line([100, nose_y, 100, nose_y + 20], fill=(0, 0, 0), width=2)
        
        # Draw mouth (wider for female)
        mouth_y = 130 + random.randint(-5, 5)
        mouth_width = 50 if gender == 1 else 40
        draw.arc([100 - mouth_width//2, mouth_y, 100 + mouth_width//2, mouth_y + 20], 
                 start=0, end=180, fill=(0, 0, 0), width=2)
        
        # Add age-related features
        if age > 40:
            # Add wrinkles
            draw.line([80, 90, 85, 92], fill=(100, 100, 100), width=1)
            draw.line([115, 92, 120, 90], fill=(100, 100, 100), width=1)
        
        # Add random variations (lighting, orientation)
        pixels = np.array(img)
        noise = np.random.normal(0, 10, pixels.shape).astype(np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(pixels)
    
    def generate_dataset(self, num_samples=1000):
        """Generate synthetic dataset with UTKFace naming convention"""
        print(f"🎨 Generating {num_samples} synthetic face images...")
        print(f"📁 Output: {self.output_dir}")
        
        for i in range(num_samples):
            # Random attributes
            age = random.randint(1, 116)
            gender = random.randint(0, 1)  # 0=male, 1=female
            race = random.randint(0, 4)    # 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
            
            # Generate image
            img = self.generate_face_image(age, gender, race)
            
            # UTKFace format: [age]_[gender]_[race]_[timestamp].jpg
            filename = f"{age}_{gender}_{race}_{i:08d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            img.save(filepath, quality=85)
            
            if (i + 1) % 100 == 0:
                print(f"  ✓ Generated {i + 1}/{num_samples} images")
        
        print(f"\n✅ Successfully generated {num_samples} synthetic faces!")
        print(f"📊 Distribution:")
        print(f"   Ages: 1-116 years")
        print(f"   Genders: Male (0), Female (1)")
        print(f"   Races: 5 categories")
        return self.output_dir

if __name__ == '__main__':
    generator = SyntheticFaceGenerator()
    generator.generate_dataset(num_samples=1000)
