import torch
import torch.nn as nn
from torchvision import models, transforms
import os
from PIL import Image

def run_prediction():

    FILENAME = "archiargiolestes-parvulus-female-wings_34787831906_o.jpg"
    

    print("Яку модель використовуємо для прогнозу?\n1 - Сиру\n2 - Оброблених скелетів")
    choice = input("Вибір: ")
    
    output_dir = 'res_train'
    test_dir = 'test_files'
    

    target_folders = ['type1', 'type2'] if choice == '1' else ['type1_processed', 'type2_processed']
    model_filename = 'wing_model_raw.pth' if choice == '1' else 'wing_model_processed.pth'
    
    model_path = os.path.join(output_dir, model_filename)
    image_path = os.path.join(test_dir, FILENAME)


    if not os.path.exists(model_path):
        print(f"Помилка: Модель {model_path} не знайдена!"); return
    if not os.path.exists(image_path):
        print(f"Помилка: Файл {image_path} не знайдено!"); return

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

    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)

    print(f"\n--- Аналіз файлу: {FILENAME} ---")
    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        pred_name = target_folders[predicted.item()]
        conf_percent = confidence.item() * 100

    print(f"Результат: {pred_name}")
    print(f"Впевненість моделі: {conf_percent:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    run_prediction()