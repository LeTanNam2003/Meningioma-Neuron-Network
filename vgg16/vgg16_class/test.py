import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.conv(x)

class BatchNorm2D(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm2D, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        return self.bn(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc(x)

class VGG16_Custom(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16_Custom, self).__init__()
        
        self.features = nn.Sequential(
            Conv2D(3, 64, kernel_size=3, padding=1), BatchNorm2D(64), SiLU(),
            Conv2D(64, 64, kernel_size=3, padding=1), BatchNorm2D(64), SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv2D(64, 128, kernel_size=3, padding=1), BatchNorm2D(128), SiLU(),
            Conv2D(128, 128, kernel_size=3, padding=1), BatchNorm2D(128), SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv2D(128, 256, kernel_size=3, padding=1), BatchNorm2D(256), SiLU(),
            Conv2D(256, 256, kernel_size=3, padding=1), BatchNorm2D(256), SiLU(),
            Conv2D(256, 256, kernel_size=3, padding=1), BatchNorm2D(256), SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv2D(256, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            Linear(512 * 7 * 7, 4096), SiLU(),
            Linear(4096, 4096), SiLU(),
            Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.classifier(x)
        return x

# Tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = 1 if "tumor" in self.image_files[idx].lower() else 0
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# Load tập dữ liệu
train_dataset = BrainTumorDataset("C:/Personal/final_graduate/data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình
model = VGG16_Custom(num_classes=2).to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Vòng lặp huấn luyện
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def evaluate(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

test_dataset = BrainTumorDataset("C:/Personal/final_graduate/data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
evaluate(model, test_loader)