import sys
import os
import argparse
import subprocess
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).parent / "process_image_src"))

from inference import predict_one_image
from train import run_training
from process_images import process_image_binary
from get_binary_images import create_mask, BinaryImage

def main():
    parser = argparse.ArgumentParser(description="Insect Wing AI Pipeline")
    parser.add_argument("image", help="Path to the image to analyze")
    parser.add_argument("--retrain", action="store_true", help="Force retraining the model")
    parser.add_argument("--reprocess", action="store_true", help="Force re-processing of all images via C++")
    
    args = parser.parse_args()
    img_path = Path(args.image)
    model_path = Path("res_train/wing_model_processed.pth")
    cpp_script = Path("process_image_src/main.py")

    if args.reprocess:
        print("Forced reprocess: Running C++ batch processor...")
        subprocess.run([sys.executable, str(cpp_script)], check=True)
        args.retrain = True 

    if args.retrain or not model_path.exists():
        if model_path.exists(): model_path.unlink()
        print("Running training pipeline...")
        run_training(force_choice='2')

    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None: return print(f"Error: Could not load {img_path}")
    
    binary_pixels = create_mask(img_gray)
    item = BinaryImage(binary_pixels, img_path, status="preprocess_binary")

    process_image_binary(item, Path("."), radius=1, threshold=100)

    classes = ["type1", "type2"] 
    label, conf = predict_one_image(item.pixels, str(model_path), classes)

    print(f"Species identified: {label}")

if __name__ == "__main__":
    main()
