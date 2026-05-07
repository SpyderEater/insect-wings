import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

def predict_one_image(pixel_array, model_path, target_folders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(target_folders))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.fromarray(pixel_array).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return target_folders[predicted.item()], confidence.item() * 100
