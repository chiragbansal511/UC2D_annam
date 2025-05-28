#  Soil Type Classification using Deep Learning

This project focuses on **classifying soil types from images** using advanced deep learning techniques. It explores the use of **custom Convolutional Neural Networks (CNNs)** as well as **transfer learning with MobileNetV2 and ResNet50**, coupled with ensemble learning strategies to maximize classification performance.

The models are trained to identify **four types of soil**:

-  Alluvial Soil  
-  Black Soil  
-  Clay Soil  
-  Red Soil

---

##  Project Highlights

### ✅ Features

- **Multiple Model Architectures**
  - Custom-built CNN
  - MobileNetV2 (pre-trained on ImageNet)
  - ResNet50 (pre-trained on ImageNet)

- **Transfer Learning**
  - Leveraging pretrained weights for faster convergence and better generalization.

- **Data Augmentation**
  - `main.ipynb`: Standard augmentations – Horizontal Flip, Rotation, Color Jitter.
  - `test.ipynb`: Advanced augmentations – Resized Crop, Vertical Flip, Gaussian Blur, Erasing.

- **Ensemble Learning**
  - `main.ipynb`: Simple average of softmax outputs.
  - `test.ipynb`: Weighted ensemble for improved accuracy.

- **Training Optimizations**
  - Mixed-precision training using `torch.cuda.amp`
  - Label smoothing for improved generalization
  - Cosine annealing scheduler
  - Early stopping to prevent overfitting

- **Hardware Support**
  - GPU acceleration (CUDA)
  - CPU-only inference script provided

---

