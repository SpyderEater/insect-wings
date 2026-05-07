Insect Wing Classification Pipeline
This project provides an automated end-to-end pipeline for processing insect wing images using a custom C++/TBB engine and classifying them with a ResNet18 neural network.  

###Prerequisites
Python 3.12+
C++ Compiler (with Intel TBB support)
Libraries: torch, torchvision, numpy<2, opencv-python, pillow

###Usage

```bash
cd process_image_src
python3 setup.py build_ext --inplace
cd ..
```

```bash
python3 main.py path/to/wing_image.png
```

###Advanced Commands
Force Reprocess All: Run the C++ batch processor on the entire input_images folder and then retrain the AI.

```bash
python3 main.py path/to/image.png --reprocess
```


Force Retrain: Keep existing skeletons but retrain the ResNet18 model.

```bash
python3 main.py path/to/image.png --retrain
```

Project Structure

main.py: The global entry point and orchestrator.  

train.py: Handles automated ResNet18 training with dynamic class detection.

inference.py: Logic for neural network predictions using processed skeletons.

process_image_src/: Core C++ implementation using Intel TBB for venation extraction.  

input_images/: Raw insect wing scans organized by subfolders (species).

output_images/: Generated binary masks (skeletons) produced by the C++ engine.

res_train/: Saved weights (.pth) and class labels.

Methodology
The system extracts high-fidelity binary masks using an adaptive thresholding approach followed by a custom median filter implemented in C++. This "skeleton" is then used as input for a ResNet18 classifier to ensure the model focuses on venation patterns rather than background noise.
