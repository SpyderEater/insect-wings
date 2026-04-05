""" main """

import sys
from pathlib import Path
from PIL import Image
from get_binary_images import get_binary_dataset
from process_images import process_image_binary

def main_debug(item, root, output_dir):
    original_pixels = item.pixels.copy()
    
    test_cases = [
        # Радіус 1 (вікно 3х3) - швидкий, зберігає дрібні деталі
        (1, 20), (1, 60), (1, 80), (1, 100)
        
        # Радіус 2 (вікно 5х5) - краще прибирає шум, але може "з'їсти" кінчики жилок
        , (2, 50), (2, 70), (2, 90), (2, 110), (2, 130), (2, 160), (4, 200)
    ]
    
    print(f"--- RUNNING DEBUG: {item.relative_path} ---")
    
    for r, t in test_cases:
        item.pixels = original_pixels.copy()
        item.status = "preprocess_binary"
        
        process_image_binary(item, root, radius=r, threshold=t, is_debug=True)
        
        debug_name = f"{item.relative_path.stem}_R{r}_T{t}{item.relative_path.suffix}"
        debug_path = output_dir / "debug" / debug_name
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        
        Image.fromarray(item.pixels).save(debug_path)
        print(f"Saved variant: R={r}, T={t}")

def main():
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "input_images"
    output_dir = root / "output_images"
    
    dataset = get_binary_dataset(input_dir)
    
    # Перевірка аргументів терміналу
    is_debug_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "debug"

    for item in dataset.items:
        if is_debug_mode:
            main_debug(item, root, output_dir)
            return
        else:
            process_image_binary(item, root, radius=3, threshold=80, is_debug=False)
            
            if item.status != "processed_binary":
                raise ValueError(f"Об'єкт {item.relative_path} має статус {item.status}")
            
            one_img_output_path = output_dir / item.relative_path
            one_img_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(item.pixels).save(one_img_output_path)
            
            item.status = "processed_image"
            print(f"Status: {item.status} | {item.relative_path}")

if __name__ == "__main__":
    main()
