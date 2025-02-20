import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.biases = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return F.conv2d(x, self.weights, self.biases, stride=self.stride, padding=self.padding)

class BatchNorm2D:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = torch.nn.Parameter(torch.ones(num_features))
        self.beta = torch.nn.Parameter(torch.zeros(num_features))
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, training=True):
        if training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.squeeze()
        else:
            x_norm = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)

        return self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)

class SiLU:
    def forward(self, x):
        return x * torch.sigmoid(x)

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size  # Nếu stride không có, lấy bằng kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        output = torch.zeros(batch_size, channels, out_height, out_width)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        region = x[b, c, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                        output[b, c, i, j] = torch.max(region)

        return output

class AdaptiveAvgPool2D:
    def __init__(self, output_size):
        self.output_size = output_size  # Kích thước đầu ra mong muốn (h, w)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        stride_h = height // self.output_size[0]
        stride_w = width // self.output_size[1]

        output = torch.zeros(batch_size, channels, self.output_size[0], self.output_size[1])
        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.output_size[0]):
                    for j in range(self.output_size[1]):
                        region = x[b, c, i*stride_h:(i+1)*stride_h, j*stride_w:(j+1)*stride_w]
                        output[b, c, i, j] = torch.mean(region)

        return output

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.biases = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weights.T) + self.biases

class VGG16_Custom:
    def __init__(self, num_classes=1000):
        super(VGG16_Custom, self).__init__()

        self.features = [
            # Block 1
            Conv2D(3, 64, kernel_size=3, padding=1), BatchNorm2D(64), SiLU(),
            Conv2D(64, 64, kernel_size=3, padding=1), BatchNorm2D(64), SiLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Block 2
            Conv2D(64, 128, kernel_size=3, padding=1), BatchNorm2D(128), SiLU(),
            Conv2D(128, 128, kernel_size=3, padding=1), BatchNorm2D(128), SiLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Block 3
            Conv2D(128, 256, kernel_size=3, padding=1), BatchNorm2D(256), SiLU(),
            Conv2D(256, 256, kernel_size=3, padding=1), BatchNorm2D(256), SiLU(),
            Conv2D(256, 256, kernel_size=3, padding=1), BatchNorm2D(256), SiLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Block 4
            Conv2D(256, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Block 5
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            Conv2D(512, 512, kernel_size=3, padding=1), BatchNorm2D(512), SiLU(),
            MaxPool2D(kernel_size=2, stride=2),

            # Adaptive AvgPool2D
            AdaptiveAvgPool2D((7, 7))
        ]

        self.classifier = [
            Linear(512 * 7 * 7, 4096), SiLU(),
            Linear(4096, 4096), SiLU(),
            Linear(4096, num_classes)
        ]

    def forward(self, x):
        for layer in self.features:
            x = layer.forward(x)

        x = x.view(x.shape[0], -1)  # Flatten
        for layer in self.classifier:
            x = layer.forward(x)

        return x

# Định nghĩa tiền xử lý ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về 224x224 (chuẩn VGG16)
    transforms.ToTensor()  # Chuyển đổi ảnh thành tensor
])

# Dataset tùy chỉnh
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
        label = 1 if "tumor" in img_path else 0  # Giả sử tên file chứa "tumor" nếu có khối u

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Load tập dữ liệu
train_dataset = BrainTumorDataset("C:/Personal/final_graduate/data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo mô hình
model = VGG16_Custom(num_classes=2)  # 2 lớp (có u màng não hoặc không)
model.to(device)

# Loss function và optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [param for layer in model.features + model.classifier for param in layer.parameters() if isinstance(param, torch.nn.Parameter)], 
    lr=0.001
)

# Vòng lặp huấn luyện
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model.forward(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Cập nhật trọng số
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

test_dataset = BrainTumorDataset("C:/Personal/final_graduate/data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
evaluate(model, test_loader)