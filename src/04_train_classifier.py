import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# === Konfiguration ===
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_PATH = "synth_classifier_cpp.pt"
USE_TORCHSCRIPT = True  # Für C++ Export

# === Feature-Normalisierung ===
def normalize_features(X, mean=None, std=None):
    mean = np.mean(X, axis=0) if mean is None else mean
    std = np.std(X, axis=0) if std is None else std
    std[std == 0] = 1e-8
    return (X - mean) / std, mean, std

# === Datensatz laden ===
def load_dataset(split, mean=None, std=None):
    X_path, y_path = f"data/{split}_features.npy", f"data/{split}_labels.npy"
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"{split} Dateien nicht gefunden!")

    X = np.load(X_path)
    y = np.load(y_path)

    X, mean_out, std_out = normalize_features(X, mean, std)
    return TensorDataset(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.long)), mean_out, std_out

# === Daten vorbereiten ===
train_dataset, mean, std = load_dataset("train")
val_dataset, _, _ = load_dataset("val", mean, std)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === Modell ===
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
FEATURE_DIM = train_dataset.tensors[0].shape[1]
NUM_CLASSES = len(torch.unique(train_dataset.tensors[1]))

model = Classifier(FEATURE_DIM, NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# === Training ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # === Validierung ===
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)

            val_loss += loss.item() * X_val.size(0)
            val_correct += (outputs.argmax(1) == y_val).sum().item()
            val_total += y_val.size(0)

    scheduler.step()

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# === Modell speichern ===
torch.save(model.state_dict(), MODEL_PATH)
print(f"Modell gespeichert als {MODEL_PATH}")

# === Optional: TorchScript (für C++ / Deployment) ===
if USE_TORCHSCRIPT:
    example_input = torch.randn(1, FEATURE_DIM).to(device)
    traced = torch.jit.trace(model, example_input)
    traced.save("synth_classifier_cpp_traced.pt")
    print("TorchScript-Modell für C++ gespeichert als synth_classifier_cpp_traced.pt")
