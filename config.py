# config.py

import torch
import torchvision.transforms as transforms
import os

# --- General Settings ---

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = 'saidas'
LOG_FILE = 'execution_log.txt'
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'saved_models')

# --- Dataset Configuration ---
# To activate a dataset, add its name to the list.
# The key must match the loading function in data_loader.py
ACTIVE_DATASETS = {
    'RAF-DB': 'load_rafdb',
    'ExpW': 'load_expw',
    'FER2013': 'load_fer2013',
}

# --- Model & Training Configuration ---
# Define models to train and their specific batch sizes.
# This is where you optimize GPU usage per model.
MODEL_CONFIG = {
    'densenet121': {'batch_size': 96},
    'resnet50': {'batch_size': 128},
    'efficientnet_b0': {'batch_size': 128},
    # 'convnext_tiny': {'batch_size': 128}, # Example for adding more
}

# --- Hyperparameters ---
EPOCHS = 100 # high number for early stopping
LEARNING_RATE = 0.001
PATIENCE = 5 # For early stopping

# --- Data Augmentation Transforms ---
TRANSFORM_TRAIN_HEAVY = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_tta_transforms():
    return [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(200),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]
