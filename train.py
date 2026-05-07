import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class InsectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        self.classes = sorted([d for d in os.listdir(root_dir) 
                             if os.path.isdir(os.path.join(root_dir, d)) and d != 'debug'])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        for target_class in self.classes:
            class_path = os.path.join(root_dir, target_class)
            for filename in os.listdir(class_path):
                if filename.lower().endswith(valid_extensions):
                    path = os.path.join(class_path, filename)
                    self.samples.append((path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def run_training(force_choice='2'):
    output_dir = 'res_train'
    os.makedirs(output_dir, exist_ok=True)

    base_data_dir = 'input_images' if force_choice == '1' else 'output_images'
    model_filename = 'wing_model_raw.pth' if force_choice == '1' else 'wing_model_processed.pth'
    model_path = os.path.join(output_dir, model_filename)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(base_data_dir):
        return

    dataset = InsectDataset(base_data_dir, transform=transform)
    num_classes = len(dataset.classes)
    
    if num_classes == 0:
        return

    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        f.write("\n".join(dataset.classes))

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    run_training()