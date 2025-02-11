import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from torchvision import models
from torchsummary import summary

# Load trained model
class BrainTumorVGG16(nn.Module):
    def __init__(self, num_classes=1):
        super(BrainTumorVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)  # Load pre-trained weights
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.vgg16(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorVGG16().to(device)
summary(model, (3, 224, 224))
model.load_state_dict(torch.load('C:/Personal/final_graduate/commit2/Meningioma_Model/v4/best_model_vgg16_v4.pth', map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Đảm bảo đầu vào có 3 kênh như khi train
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def segment_tumor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def predict_and_segment(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale (MRI thường 1 kênh)
    image = image.convert('RGB')  # Chuyển thành 3 kênh để khớp mô hình
    model_input = transform(image).unsqueeze(0).to(device)
    output = model(model_input)
    prediction = torch.sigmoid(output).item()
    
    if prediction < 0.5:  # Có khối u
        segmented_image = segment_tumor(np.array(image))
        return True, segmented_image
    else:  # Không có khối u
        return False, None

# GUI
class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection")
        
        self.label = tk.Label(root, text="Chọn ảnh MRI để kiểm tra u màng não")
        self.label.pack()
        
        self.btn_select = tk.Button(root, text="Chọn ảnh", command=self.load_image)
        self.btn_select.pack()
        
        self.canvas_original = tk.Label(root)
        self.canvas_original.pack()
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack()
        
        self.canvas_segmented = tk.Label(root)
        self.canvas_segmented.pack()
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = Image.open(file_path).convert('RGB')
            image.thumbnail((300, 300))
            self.photo = ImageTk.PhotoImage(image)
            self.canvas_original.config(image=self.photo)
            
            has_tumor, segmented_image = predict_and_segment(file_path)
            
            if has_tumor:
                self.result_label.config(text="Phát hiện u màng não", fg="red")
                segmented_image.thumbnail((300, 300))
                self.segmented_photo = ImageTk.PhotoImage(segmented_image)
                self.canvas_segmented.config(image=self.segmented_photo)
            else:
                self.result_label.config(text="Không phát hiện u màng não", fg="green")
                self.canvas_segmented.config(image='')  # Xóa ảnh phân đoạn nếu không có khối u

# Run GUI
root = tk.Tk()
app = TumorDetectionApp(root)
root.mainloop()
