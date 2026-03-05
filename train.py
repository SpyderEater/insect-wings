import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class InsectDataset(Dataset):
    def __init__(self, target_folders, transform=None):
        self.samples = []
        self.transform = transform
        
        self.class_to_idx = {folder: i for i, folder in enumerate(target_folders)}
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for folder in target_folders:
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                if filename.lower().endswith(valid_extensions):
                    path = os.path.join(folder, filename)
                    self.samples.append((path, self.class_to_idx[folder]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def run_training():
    output_dir = 'res_train'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Оберіть дані для навчання:\n1 - Сирі (type1, type2)\n2 - Скелети (type1_processed, type2_processed)")
    choice = input("Ваш вибір: ")

    target_folders = ['type1', 'type2'] if choice == '1' else ['type1_processed', 'type2_processed']
    model_filename = 'wing_model_raw.pth' if choice == '1' else 'wing_model_processed.pth'
    model_path = os.path.join(output_dir, model_filename)

    for f in target_folders:
        if not os.path.exists(f):
            print(f"Помилка: Папка {f} відсутня!"); return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = InsectDataset(target_folders, transform=transform)
    
    if len(dataset) == 0:
        print("Помилка: У вибраних папках не знайдено зображень!"); return

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(target_folders))
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    print(f"Початок навчання на {len(dataset)} фото. Результат буде в {output_dir}...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Епоха {epoch+1}/10 | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Модель збережена: {model_path}")

if __name__ == "__main__":
    run_training()