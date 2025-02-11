import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2

# Data Preprocessing and Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Loại bỏ ColorJitter vì không phù hợp với MRI
])

# Load the dataset from directory
dataset = datasets.ImageFolder('C:/Personal/final_graduate/data', transform=transform)

# Split dataset into training and validation (80% - 20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):  # Define the number of output classes
        super(VGG16, self).__init__()

        # Feature extractor (Convolutional layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 -> 7
            
            # AdaptiveAvgPool2d added here
            nn.AdaptiveAvgPool2d((7, 7))  # Ensures fixed 7x7 output
        )

        # Fully Connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # No Softmax for BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch_size, 512*7*7)
        x = self.classifier(x)
        return x

# Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=1).to(device)

# Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Thay CrossEntropyLoss bằng BCEWithLogitsLoss
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Giảm learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Training Settings
num_epochs = 30
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # Định dạng label thành (batch, 1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        predictions = (torch.sigmoid(outputs) > 0.5).float()  # Chuyển sang nhị phân
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    
    scheduler.step(val_loss)  # Điều chỉnh learning rate khi cần
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_model_vgg16_v4_4.pth')
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

# Load the Best Model
model.load_state_dict(torch.load('best_model_vgg16_v4_4.pth'))
model.eval()

# Apply segmentation to new images
def segment_tumor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Thay bằng Otsu thresholding
    return binary

def predict_and_segment(image):
    model_input = transform(image).unsqueeze(0).to(device)
    output = model(model_input)
    prediction = torch.sigmoid(output).item()
    
    if prediction > 0.5:  # Nếu có u màng não
        segmented_image = segment_tumor(np.array(image))
        return segmented_image
    else:
        return None
