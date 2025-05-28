import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.config import *
from src.dataset import SoilDataset
from src.transforms import train_transform, val_transform
from src.models import CustomCNN, get_model
from src.train import train_model
from src.predict import predict_test

# Load data
train_df = pd.read_csv(TRAIN_CSV)
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)
train_dataset = SoilDataset(train_data, TRAIN_DIR, transform=train_transform)
val_dataset = SoilDataset(val_data, TRAIN_DIR, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Train
model1 = train_model(CustomCNN(), "CustomCNN", train_loader, val_loader)
model2 = train_model(get_model("mobilenet"), "MobileNetV2", train_loader, val_loader)
model3 = train_model(get_model("resnet"), "ResNet50", train_loader, val_loader)

# Load checkpoints
model1.load_state_dict(torch.load("CustomCNN_best.pth"))
model2.load_state_dict(torch.load("MobileNetV2_best.pth"))
model3.load_state_dict(torch.load("ResNet50_best.pth"))

# Predict
predict_test([model1.to(device), model2.to(device), model3.to(device)], TEST_DIR, TEST_CSV, SUBMISSION_CSV)