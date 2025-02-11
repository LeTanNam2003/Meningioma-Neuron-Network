import torch
import torch.nn as nn
from torchsummary import summary
import io
import sys

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

# Create and test the model
model = VGG16(num_classes=1)  # If binary classification, set num_classes=1

print(model)

# Print model summary
summary(model, (3, 224, 224))

# Capture the summary output
summary_str = io.StringIO()
sys.stdout = summary_str  # Redirect stdout to StringIO
summary(model, (3, 224, 224))  # Call summary
sys.stdout = sys.__stdout__  # Reset stdout to default

# Save summary to file
with open("vgg16_adaptive.txt", "w") as f:
    f.write(summary_str.getvalue())

print("Model summary saved to vgg16_adaptive.txt!")

