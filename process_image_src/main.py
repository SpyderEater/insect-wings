""" main """

import os
import time
import sys
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image
from get_binary_images import get_binary_dataset
from process_images import process_image_binary



def compile_cpp_module():
    root = Path(__file__).resolve().parent
    print("--- [BUILD] Starting C++ compilation ---")
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=root,
        check=True 
    )
    print("--- [BUILD] Finished successfully ---")

def main_debug(item, root, output_dir):
    from process_images import process_image_binary
    original_pixels = item.pixels.copy()
    
    test_cases = [
        (1, 20), (1, 60), (1, 80), (1, 100)
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

def process_item(item, root):
    from process_images import process_image_binary
    
    pid = os.getpid()
    print(f"--- [Process {pid}] Started: {item.relative_path.name}")
    
    item.status = "preprocess_binary"
    process_image_binary(item, root, radius=1, threshold=100, is_debug=False)
    
    return (item.pixels, item.relative_path, pid)

def main():
    args = [a.lower() for a in sys.argv]
    if "build" in args:
        compile_cpp_module()

    
    root = Path(__file__).resolve().parent.parent
    input_dir = root / "input_images"
    output_dir = root / "output_images"
    
    dataset = get_binary_dataset(input_dir)
    
    is_debug_mode = "debug" in args

    if is_debug_mode:
        for item in dataset.items:
            main_debug(item, root, output_dir)
        return

    args_list = [(item, root) for item in dataset.items]

    n_threads_input = input(f"Enter number of processes (default {cpu_count()}): ").strip()
    n = int(n_threads_input) if n_threads_input else cpu_count()

    print(f"Total images: {len(dataset.items)}")

    start = time.perf_counter()

    with Pool(n) as pool:
        for i, (pixels, relative_path, pid) in enumerate(pool.starmap(process_item, args_list)):
            
            one_img_output_path = output_dir / relative_path
            one_img_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            Image.fromarray(pixels).save(one_img_output_path)
            print(f"[{i+1}/{len(args_list)}] Done: {relative_path} (by Process {pid})")
        
    print(f"Total images: {len(dataset.items)}")

    end = time.perf_counter()
    print(f"Processing time: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()