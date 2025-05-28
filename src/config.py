import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = 'D:/Projects/iit_ropar_project/train'
TEST_DIR = 'D:/Projects/iit_ropar_project/test'
TRAIN_CSV = 'D:/Projects/iit_ropar_project/train_labels.csv'
TEST_CSV = 'D:/Projects/iit_ropar_project/test_ids.csv'
SUBMISSION_CSV = 'D:/Projects/iit_ropar_project/sample_submission.csv'

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 30
LABELS = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']
label2idx = {label: i for i, label in enumerate(LABELS)}
idx2label = {i: label for label, i in label2idx.items()}
