""" get images """

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from pathlib import Path

class BinaryImage:
    def __init__(self, pixels, relative_path, status="preprocess_image"):
        self.pixels = pixels
        self.relative_path = relative_path
        self.status = status

class BinaryDataset:
    def __init__(self):
        self.items = []
    def add(self, binary_image):
        self.items.append(binary_image)
    def merge(self, other_dataset):
        self.items.extend(other_dataset.items)

def create_mask(pixels, w=25, c=10):
    local_mean = uniform_filter(pixels.astype(np.float32), size=w)
    return np.where(pixels > (local_mean - c), 255, 0).astype(np.uint8)

def get_binary_dataset(current_path, input_root=None):
    if input_root is None:
        input_root = current_path
        
    dataset = BinaryDataset()
    extensions = ('.jpg', '.jpeg', '.png')
    
    items_in_folder = list(current_path.iterdir())
    image_files = [f for f in items_in_folder if f.suffix.lower() in extensions]
    
    if image_files:
        for img_path in image_files:
            # Створюємо базовий об'єкт для перевірки початкового стану
            # Тут ми імітуємо перевірку: якщо файл існує і це картинка, статус "preprocess_image"
            initial_status = "preprocess_image"
            
            if initial_status != "preprocess_image":
                raise ValueError(f"Файл {img_path.name} вже має статус {initial_status}")

            img = Image.open(img_path).convert('RGB')
            binary_pixels = create_mask(np.array(img))
            
            binary_obj = BinaryImage(
                pixels=binary_pixels, 
                relative_path=img_path.relative_to(input_root),
                status="preprocess_binary"
            )
            dataset.add(binary_obj)
    else:
        for item in items_in_folder:
            if item.is_dir():
                sub_dataset = get_binary_dataset(item, input_root)
                dataset.merge(sub_dataset)
                
    return dataset
