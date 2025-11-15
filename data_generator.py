"""
Synthetic Data Generator for Vision-Based Preventive Maintenance
Creates realistic machine part images with various defect types
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import DATASET_CONFIG, DEFECT_CONFIG


class SyntheticDataGenerator:
    def __init__(self, config=DATASET_CONFIG, defect_config=DEFECT_CONFIG):
        self.config = config
        self.defect_config = defect_config
        self.image_size = config['image_size']
        np.random.seed(config['seed'])
        random.seed(config['seed'])
    
    def generate_base_part(self):
        """Generate a base machine part image (circular/rectangular component)"""
        img = Image.new('RGB', self.image_size, color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        
        # Random part shape
        part_type = random.choice(['circular', 'rectangular', 'complex'])
        
        if part_type == 'circular':
            # Circular part (bearing, gear, etc.)
            center = (self.image_size[0]//2, self.image_size[1]//2)
            radius = random.randint(30, 50)
            
            # Main body
            draw.ellipse([center[0]-radius, center[1]-radius, 
                         center[0]+radius, center[1]+radius], 
                        fill=(150, 150, 150), outline=(100, 100, 100), width=2)
            
            # Inner circle
            inner_radius = radius // 3
            draw.ellipse([center[0]-inner_radius, center[1]-inner_radius,
                         center[0]+inner_radius, center[1]+inner_radius],
                        fill=(80, 80, 80))
            
        elif part_type == 'rectangular':
            # Rectangular part (plate, bracket, etc.)
            width = random.randint(60, 90)
            height = random.randint(40, 70)
            x = (self.image_size[0] - width) // 2
            y = (self.image_size[1] - height) // 2
            
            draw.rectangle([x, y, x+width, y+height], 
                          fill=(160, 160, 160), outline=(100, 100, 100), width=2)
            
            # Add some details (holes, edges)
            for _ in range(random.randint(2, 4)):
                hole_x = random.randint(x+10, x+width-10)
                hole_y = random.randint(y+10, y+height-10)
                hole_radius = random.randint(3, 8)
                draw.ellipse([hole_x-hole_radius, hole_y-hole_radius,
                             hole_x+hole_radius, hole_y+hole_radius],
                            fill=(50, 50, 50))
        
        else:  # complex
            # Complex part with multiple features
            center = (self.image_size[0]//2, self.image_size[1]//2)
            
            # Main body
            draw.rectangle([center[0]-40, center[1]-30, center[0]+40, center[1]+30],
                          fill=(140, 140, 140), outline=(100, 100, 100), width=2)
            
            # Extensions
            draw.rectangle([center[0]-50, center[1]-10, center[0]-40, center[1]+10],
                          fill=(140, 140, 140))
            draw.rectangle([center[0]+40, center[1]-10, center[0]+50, center[1]+10],
                          fill=(140, 140, 140))
        
        # Add surface texture
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def add_scratch_defect(self, img):
        """Add scratch defects to the image"""
        img_array = np.array(img)
        draw = ImageDraw.Draw(img)
        
        num_scratches = random.randint(1, 3)
        for _ in range(num_scratches):
            # Random scratch parameters
            start_x = random.randint(20, self.image_size[0]-20)
            start_y = random.randint(20, self.image_size[1]-20)
            length = random.randint(15, 40)
            angle = random.uniform(0, 2*np.pi)
            
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            # Ensure end point is within image
            end_x = max(0, min(self.image_size[0]-1, end_x))
            end_y = max(0, min(self.image_size[1]-1, end_y))
            
            # Draw scratch
            draw.line([(start_x, start_y), (end_x, end_y)], 
                     fill=(50, 50, 50), width=random.randint(1, 3))
        
        return img
    
    def add_crack_defect(self, img):
        """Add crack defects to the image"""
        draw = ImageDraw.Draw(img)
        
        # Main crack
        start_x = random.randint(10, self.image_size[0]-10)
        start_y = random.randint(10, self.image_size[1]-10)
        
        # Create branching crack pattern
        current_x, current_y = start_x, start_y
        angle = random.uniform(0, 2*np.pi)
        
        for _ in range(random.randint(3, 8)):
            length = random.randint(8, 20)
            angle += random.uniform(-0.5, 0.5)  # Slight direction change
            
            next_x = int(current_x + length * np.cos(angle))
            next_y = int(current_y + length * np.sin(angle))
            
            # Ensure within bounds
            next_x = max(5, min(self.image_size[0]-5, next_x))
            next_y = max(5, min(self.image_size[1]-5, next_y))
            
            draw.line([(current_x, current_y), (next_x, next_y)],
                     fill=(30, 30, 30), width=1)
            
            current_x, current_y = next_x, next_y
            
            # Random branch
            if random.random() < 0.3:
                branch_angle = angle + random.uniform(-np.pi/3, np.pi/3)
                branch_length = random.randint(5, 15)
                branch_x = int(current_x + branch_length * np.cos(branch_angle))
                branch_y = int(current_y + branch_length * np.sin(branch_angle))
                
                branch_x = max(5, min(self.image_size[0]-5, branch_x))
                branch_y = max(5, min(self.image_size[1]-5, branch_y))
                
                draw.line([(current_x, current_y), (branch_x, branch_y)],
                         fill=(30, 30, 30), width=1)
        
        return img
    
    def add_corrosion_defect(self, img):
        """Add corrosion/rust defects to the image"""
        img_array = np.array(img)
        
        num_spots = random.randint(2, 5)
        for _ in range(num_spots):
            center_x = random.randint(20, self.image_size[0]-20)
            center_y = random.randint(20, self.image_size[1]-20)
            radius = random.randint(8, 20)
            
            # Create irregular corrosion shape
            y, x = np.ogrid[:self.image_size[1], :self.image_size[0]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Add noise to make it irregular
            noise_mask = np.random.random((self.image_size[1], self.image_size[0])) > 0.3
            mask = mask & noise_mask
            
            # Apply rust color
            rust_color = np.array([139, 69, 19])  # Brown/rust color
            img_array[mask] = rust_color + np.random.randint(-20, 20, 3)
            img_array[mask] = np.clip(img_array[mask], 0, 255)
        
        return Image.fromarray(img_array)
    
    def add_dent_defect(self, img):
        """Add dent defects (simulated with shading)"""
        img_array = np.array(img).astype(np.float32)
        
        num_dents = random.randint(1, 2)
        for _ in range(num_dents):
            center_x = random.randint(25, self.image_size[0]-25)
            center_y = random.randint(25, self.image_size[1]-25)
            radius = random.randint(10, 25)
            
            # Create dent effect with gradient
            y, x = np.ogrid[:self.image_size[1], :self.image_size[0]]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Gaussian-like dent effect
            dent_mask = distance <= radius
            dent_intensity = np.exp(-(distance**2) / (2 * (radius/3)**2))
            dent_intensity[~dent_mask] = 0
            
            # Darken the area to simulate depth
            darkening = dent_intensity * 40
            img_array[dent_mask] -= darkening[dent_mask, np.newaxis]
            img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def add_stain_defect(self, img):
        """Add stain defects (oil, grease, etc.)"""
        img_array = np.array(img)
        
        num_stains = random.randint(1, 3)
        for _ in range(num_stains):
            center_x = random.randint(15, self.image_size[0]-15)
            center_y = random.randint(15, self.image_size[1]-15)
            
            # Irregular stain shape
            stain_size = random.randint(8, 18)
            stain_img = Image.new('RGBA', self.image_size, (0, 0, 0, 0))
            stain_draw = ImageDraw.Draw(stain_img)
            
            # Create irregular blob
            points = []
            for angle in np.linspace(0, 2*np.pi, 8):
                radius = stain_size + random.randint(-3, 3)
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                points.append((x, y))
            
            stain_color = (40, 40, 40, 120)  # Semi-transparent dark stain
            stain_draw.polygon(points, fill=stain_color)
            
            # Blur for realistic effect
            stain_img = stain_img.filter(ImageFilter.GaussianBlur(radius=1))
            
            # Composite with original image
            img = Image.alpha_composite(img.convert('RGBA'), stain_img).convert('RGB')
        
        return img
    
    def generate_defective_image(self):
        """Generate an image with random defects"""
        img = self.generate_base_part()
        
        # Randomly apply defects based on probabilities
        defects_applied = []
        
        if random.random() < self.defect_config['scratch_probability']:
            img = self.add_scratch_defect(img)
            defects_applied.append('scratch')
        
        if random.random() < self.defect_config['crack_probability']:
            img = self.add_crack_defect(img)
            defects_applied.append('crack')
        
        if random.random() < self.defect_config['corrosion_probability']:
            img = self.add_corrosion_defect(img)
            defects_applied.append('corrosion')
        
        if random.random() < self.defect_config['dent_probability']:
            img = self.add_dent_defect(img)
            defects_applied.append('dent')
        
        if random.random() < self.defect_config['stain_probability']:
            img = self.add_stain_defect(img)
            defects_applied.append('stain')
        
        # Ensure at least one defect is applied
        if not defects_applied:
            defect_type = random.choice(['scratch', 'crack', 'corrosion', 'dent', 'stain'])
            if defect_type == 'scratch':
                img = self.add_scratch_defect(img)
            elif defect_type == 'crack':
                img = self.add_crack_defect(img)
            elif defect_type == 'corrosion':
                img = self.add_corrosion_defect(img)
            elif defect_type == 'dent':
                img = self.add_dent_defect(img)
            elif defect_type == 'stain':
                img = self.add_stain_defect(img)
        
        return img
    
    def generate_normal_image(self):
        """Generate a normal (non-defective) image"""
        return self.generate_base_part()
    
    def generate_dataset(self):
        """Generate the complete dataset"""
        print("Generating synthetic dataset...")
        
        # Create directories
        normal_dir = os.path.join(self.config['data_dir'], 'normal')
        defective_dir = os.path.join(self.config['data_dir'], 'defective')
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(defective_dir, exist_ok=True)
        
        num_samples = self.config['num_samples_per_class']
        
        # Generate normal images
        print("Generating normal images...")
        for i in tqdm(range(num_samples)):
            img = self.generate_normal_image()
            img.save(os.path.join(normal_dir, f'normal_{i:04d}.png'))
        
        # Generate defective images
        print("Generating defective images...")
        for i in tqdm(range(num_samples)):
            img = self.generate_defective_image()
            img.save(os.path.join(defective_dir, f'defective_{i:04d}.png'))
        
        print(f"Dataset generated successfully!")
        print(f"Normal images: {num_samples}")
        print(f"Defective images: {num_samples}")
        print(f"Total images: {num_samples * 2}")
    
    def visualize_samples(self):
        """Visualize sample images from each class"""
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle('Sample Images from Dataset', fontsize=16)
        
        # Normal samples
        for i in range(4):
            img = self.generate_normal_image()
            axes[0, i].imshow(img)
            axes[0, i].set_title('Normal')
            axes[0, i].axis('off')
        
        # Defective samples
        for i in range(4):
            img = self.generate_defective_image()
            axes[1, i].imshow(img)
            axes[1, i].set_title('Defective')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join('plots', 'sample_images.png'), dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.visualize_samples()
    generator.generate_dataset()
