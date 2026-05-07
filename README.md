Insect Wing Classification Pipeline

This project provides an automated end-to-end pipeline for processing insect wing images using a custom C++/Intel TBB processing engine and classifying them with a ResNet18 neural network.

Requirements
------------
- Python 3.12+
- C++ compiler with Intel TBB support
- Python libraries:
  - torch
  - torchvision
  - numpy<2
  - opencv-python
  - pillow

Installation and Usage
------------
Build the C++ extension:

```bash
cd process_image_src
python3 setup.py build_ext --inplace
cd ..
python3 main.py images_to_test/img_1.png --reprocess
```

Structure
------------
main.py
    Main entry point and pipeline orchestrator.

train.py
    Automated ResNet18 training with dynamic class detection.

inference.py
    Neural network inference logic using processed skeleton images.

process_image_src/
    Core C++ implementation using Intel TBB for venation extraction.

input_images/
    Raw insect wing scans organized into species subfolders.

output_images/
    Generated binary masks (wing skeletons) produced by the C++ engine.

res_train/
    Saved model weights (.pth) and class label mappings.
