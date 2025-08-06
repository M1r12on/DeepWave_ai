import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd

# === Labelnamen wie im Training ===
LABELS = ["Lead", "Pluck", "Bass", "Pad"]

# === Feature-Normalisierung ===
def normalize_features(X, mean, std):
    std[std == 0] = 1e-8
    return (X - mean) / std

# === Validierungsdatensatz laden ===
def load_dataset(split, mean, std):
    X = np.load(f"data/{split}_features.npy")
    y = np.load(f"data/{split}_labels.npy")
    X = normalize_features(X, mean, std)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# === Modellklasse ===
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# === Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "synth_classifier_cpp.pt"

# === Mittelwert & Standardabweichung aus Trainingsdaten berechnen ===
X_train = np.load("data/train_features.npy")
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# === Val-Daten laden ===
X_val, y_val = load_dataset("val", mean, std)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=64)

# === Modell initialisieren und laden ===
model = Classifier(input_dim=X_val.shape[1], num_classes=len(LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Vorhersagen einsammeln ===
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.numpy())

# === Confusion Matrix ===
cm = confusion_matrix(all_targets, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – Validation Set")
plt.tight_layout()
plt.show()

# === Klassifikationsbericht ===
report_dict = classification_report(all_targets, all_preds, target_names=LABELS, output_dict=True)
df_metrics = pd.DataFrame(report_dict).transpose()
print("\n📊 Precision / Recall / F1-Score pro Klasse:\n")
print(df_metrics[["precision", "recall", "f1-score"]].round(3))
