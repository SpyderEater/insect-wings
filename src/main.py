""" main """

from multiprocessing import Pool, cpu_count

import sys
from pathlib import Path
from PIL import Image
from get_binary_images import get_binary_dataset
from process_images import process_wrapper, process_image_binary

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
    
    is_debug_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "debug"

    if is_debug_mode:
        for item in dataset.items:
            main_debug(item, root, output_dir)
        return

    # 🔥 ПАРАЛЕЛІЗАЦІЯ
    args_list = [
        (item.pixels, item.relative_path, root, 3, 80, False)
        for item in dataset.items
    ]

    print(f"Total images: {len(dataset.items)}")

    # with Pool(cpu_count()) as pool:
    #     results = pool.map(process_wrapper, args_list)

    with Pool(4) as pool:
        for i, (pixels, relative_path) in enumerate(pool.imap_unordered(process_wrapper, args_list)):
            
            one_img_output_path = output_dir / relative_path
            one_img_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(pixels).save(one_img_output_path)
            
            print(f"[{i+1}/{len(args_list)}] Done: {relative_path}")
        
        # results = pool.imap_unordered(process_wrapper, args_list)

    print(f"Total images: {len(dataset.items)}")

  

if __name__ == "__main__":
    main()
