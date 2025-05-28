import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.dataset import SoilDataset
from src.transforms import val_transform
from src.config import device, idx2label

def predict_test(models, test_dir, test_csv_path, output_csv_path):
    df = pd.read_csv(test_csv_path)
    dataset = SoilDataset(df, test_dir, labels=False, transform=val_transform)
    loader = DataLoader(dataset, batch_size=32)
    predictions = []
    ids = []
    for images, img_ids in loader:
        images = images.to(device)
        outputs = sum([model(images) for model in models]) / len(models)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = [idx2label[p] for p in preds]
        predictions.extend(labels)
        ids.extend(img_ids)
    pd.DataFrame({'image_id': ids, 'label': predictions}).to_csv(output_csv_path, index=False)