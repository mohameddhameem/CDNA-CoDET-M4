import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

CONFIG = {
    'target_column': 'model',
    'test_size': 0.2,
    'batch_size': 32,
    'epochs': 1,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

def load_dataset():
    print("Loading dataset...")
    df = pd.read_parquet('dataset/code_images.parquet')
    print(f"Loaded {len(df)} samples")

    images = []
    for img_bytes in tqdm(df['code_image'], desc="Decoding images"):
        img = pickle.loads(img_bytes)
        images.append(img)

    X = np.array(images, dtype=np.float32) / 255.0
    X = np.expand_dims(X, axis=1)
    X = np.repeat(X, 3, axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df[CONFIG['target_column']])

    return X, y, le, df

def create_model(num_classes):
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = convnext_base(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    X, y, le, df = load_dataset()

    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    print(f"Target column: {CONFIG['target_column']}")
    print(f"Classes: {le.classes_}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['seed'], stratify=y
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = create_model(num_classes).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=CONFIG['learning_rate'])

    print(f"\nTraining on {CONFIG['device']}...")
    for epoch in range(CONFIG['epochs']):
        loss = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {loss:.4f}")

    print("\nEvaluating...")
    preds, labels, probs = evaluate(model, test_loader, CONFIG['device'])

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(labels, preds)

    if num_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class='ovr')

    results_df = pd.DataFrame({
        'metric': ['accuracy', 'f1_score', 'auc'],
        'value': [accuracy, f1, auc]
    })

    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/metrics.csv', index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {CONFIG["target_column"]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nResults saved to results/")

if __name__ == "__main__":
    main()
