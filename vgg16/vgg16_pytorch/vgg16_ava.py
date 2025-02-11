import torch
import torchvision.models as models
from torchsummary import summary
import io
import sys

# Dowload model pre-trained
vgg16_pretrained = models.vgg16(pretrained=True)

# Switch the model to mode evaluate
vgg16_pretrained.eval()

# Save the model
torch.save(vgg16_pretrained.state_dict(), "vgg16_pretrained.pth")

# Dùng StringIO để bắt output của summary()
summary_str = io.StringIO()
sys.stdout = summary_str  # Chuyển stdout sang StringIO
summary(vgg16_pretrained, (3, 224, 224))  # Gọi summary
sys.stdout = sys.__stdout__  # Reset stdout về mặc định

# Ghi vào file
with open("vgg16_eva.txt", "w") as f:
    f.write(summary_str.getvalue())

