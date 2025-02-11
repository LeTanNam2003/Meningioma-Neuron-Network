# **Meningioma detection model**

Build a neuron network model can indentificate meningioma at MRI image.

**Following this step:**

_Step 1: Prepare Dataset_
- [Kangdle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Authored by Jun Cheng](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5)

_Step 2: Using available library "torch" to execute trainning the model._
| Feature                 | Model 1 (Custom VGG16) | Model 2 (Modified VGG16) | Model 3 (Pre-trained VGG16) |
|-------------------------|------------------------|--------------------------|------------------------------|
| **Data Augmentation**   | Yes (ColorJitter, Affine, etc.) | Yes (ColorJitter, Affine, etc.) | Yes (Affine, but no ColorJitter) |
| **Dataset Source**      | `'C:/Personal/final_graduate/data'` | `'C:/Personal/final_graduate/data'` | `'C:/Personal/final_graduate/data1'` |
| **Train/Test Split**    | 80% - 20%              | 80% - 20%                | 80% - 20%                     |
| **Backbone Network**    | Custom CNN (VGG-like)  | Modified VGG16           | Pre-trained VGG16             |
| **Feature Extraction**  | No (Trained from scratch) | No (Trained from scratch) | Yes (Pre-trained layers frozen) |
| **Classifier Layers**   | Fully connected (3 layers) | Fully connected (4 layers) | Fully connected (3 layers) |
| **Activation Function** | ReLU + Softmax         | ReLU + Softmax           | ReLU (no Softmax, uses raw logits) |
| **Loss Function**       | CrossEntropyLoss       | CrossEntropyLoss         | BCEWithLogitsLoss             |
| **Optimizer**          | Adam (lr=0.001)        | Adam (lr=0.001)          | Adam (lr=0.0001) + LR Scheduler |
| **Learning Rate Adjustments** | No | No | Yes (ReduceLROnPlateau) |
| **Batch Size**         | 32                     | 32                       | 32                             |
| **Training Epochs**    | 30                     | 30                       | 30                             |
| **Early Stopping**     | Yes (Patience = 5)     | Yes (Patience = 5)       | Yes (Patience = 5)             |
| **Segmentation Method** | Basic Thresholding (120) | Morphological Operations | Otsu Thresholding              |

## Summary
- **Model 1:** Custom VGG-like CNN trained from scratch.
- **Model 2:** Modified VGG16 with extra layers but still trained from scratch.
- **Model 3:** Pre-trained VGG16 with frozen feature extraction layers and a modified classifier.

### Key Differences:
- **Model 3** uses a **pre-trained VGG16** which may help with generalization.
- **Model 3** employs **BCEWithLogitsLoss** instead of CrossEntropyLoss, making it more suited for binary classification.
- **Model 3** has a **learning rate scheduler** for better optimization.
- **Model 2 & Model 3** incorporate **early stopping**, preventing overfitting.
- **Segmentation:** Model 3 applies **Otsu Thresholding**, which is more adaptive compared to the basic methods in Model 1 & 2.

_Step 3: Build a model from scratch without using predefined functions_
- Build a vgg16_adaptive model same as vgg16 model in pytorch, some different is Classifier class by config 1 output not 1000

# VGG16 Model Comparison

| **Category**               | **First Model (vgg16.txt)**    | **Second Model (vgg16_eva.txt)**   | **Third Model (vgg16_adaptive.txt)** | **Difference** |
|---------------------------|--------------------------------|------------------------------------|------------------------------------|--------------|
| **Total Parameters**      | **134,264,641**                | **138,357,544**                   | **134,264,641**                   | The second model has more parameters due to `Linear-39 (1000 outputs)` |
| **Trainable Parameters**  | 134,264,641                    | 138,357,544                        | 134,264,641                        | The second model has more trainable parameters |
| **Final Layer (Classifier Output)** | `Linear-38 [-1, 1]`  | `Linear-39 [-1, 1000]`  | `Linear-39 [-1, 1]`  | The second model is for ImageNet classification, while the other two are for binary classification |
| **Number of Fully Connected Layers** | 3                      | 3                                  | 3                                  | Same across all models |
| **Has AdaptiveAvgPool2d?** | ❌ No | ✅ Yes | ✅ Yes | The second and third models use `AdaptiveAvgPool2d` to ensure consistent input size before Fully Connected layers |
| **Total Model Size (MB)** | **731.34 MB** | **747.15 MB** | **731.53 MB** | The second model is the largest due to the 1000-class classifier |

