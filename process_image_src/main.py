import os
import time
import sys
import subprocess
from pathlib import Path
from PIL import Image
from get_binary_images import get_binary_dataset
from process_images import process_image_binary

def compile_cpp_module():
    root = Path(__file__).resolve().parent
    print("--- [BUILD] Starting C++ compilation ---")
    env = os.environ.copy()
    env["ARCHFLAGS"] = "-arch x86_64"
    
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=root,
        check=True,
        env=env
    )
    print("--- [BUILD] Finished successfully ---")

def main():
    args = [a.lower() for a in sys.argv]
    if "build" in args:
        compile_cpp_module()

    root = Path(__file__).resolve().parent.parent
    input_dir = root / "input_images"
    output_dir = root / "output_images"
    
    dataset = get_binary_dataset(input_dir)
    print(f"Total images: {len(dataset.items)}")

    start_total = time.perf_counter()

    for i, item in enumerate(dataset.items):
        relative_path = item.relative_path
        
        start_cpp = time.perf_counter()
        
        process_image_binary(item, root, radius=1, threshold=100, is_debug=False)
        
        end_cpp = time.perf_counter()
        
        one_img_output_path = output_dir / relative_path
        one_img_output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(item.pixels).save(one_img_output_path)
        
        print(f"[{i+1}/{len(dataset.items)}] CPP: {end_cpp - start_cpp:.4f}s | Done: {relative_path}")

    end_total = time.perf_counter()
    
    print("-" * 30)
    print(f"Total images processed: {len(dataset.items)}")
    print(f"Total processing time: {end_total - start_total:.2f} seconds")

if __name__ == "__main__":
    main()
