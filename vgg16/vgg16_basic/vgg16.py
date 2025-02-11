import torch
import torch.nn as nn
from torchsummary import summary
import io
import sys

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):  # Định nghĩa số lớp đầu ra
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
        )

        # Fully Connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # Không dùng softmax
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # Flatten thành (batch_size, 512*7*7)
        x = self.classifier(x)
        return x

model = VGG16(num_classes=1)  # Nếu phân loại nhị phân, để num_classes=1

print(model)

summary(model, (3, 224, 224))

# Dùng StringIO để bắt output của summary()
summary_str = io.StringIO()
sys.stdout = summary_str  # Chuyển stdout sang StringIO
summary(model, (3, 224, 224))  # Gọi summary
sys.stdout = sys.__stdout__  # Reset stdout về mặc định

# Ghi vào file
with open("vgg16.txt", "w") as f:
    f.write(summary_str.getvalue())
