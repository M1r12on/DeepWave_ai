import json
import random
from collections import defaultdict

# Eingabe- und Ausgabepfade
INPUT_FILE = "filtered_data.json"
TRAIN_FILE = "train_data.json"
VAL_FILE = "val_data.json"

# Gewünschte Verteilung pro Klasse
# Ändere hier die Verteilung pro Klasse!
SPLIT_CONFIG = {
    "Lead":    {"train": 600, "val": 150},
    "Pluck":   {"train": 600, "val": 150},
    "Bass":    {"train": 600, "val": 150},
    "Pad":     {"train": 600, "val": 150}
}

# Daten laden
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

# Nach Klassen gruppieren
by_class = defaultdict(list)
for entry in data:
    by_class[entry["label"]].append(entry)

train_set, val_set = [], []

# Split durchführen
for label, entries in by_class.items():
    random.shuffle(entries)

    n_train = SPLIT_CONFIG[label]["train"]
    n_val = SPLIT_CONFIG[label]["val"]

    if len(entries) < n_train + n_val:
        print(f"Nicht genug Daten für Klasse '{label}' – nur {len(entries)} verfügbar.")
        continue

    train_set += entries[:n_train]
    val_set += entries[n_train:n_train + n_val]

# Speichern
with open(TRAIN_FILE, "w") as f:
    json.dump(train_set, f, indent=2)
with open(VAL_FILE, "w") as f:
    json.dump(val_set, f, indent=2)

# Überblick
print("Datenaufteilung abgeschlossen:")
for label in SPLIT_CONFIG.keys():
    print(f"🔹 {label:<10} → Train: {SPLIT_CONFIG[label]['train']}, Val: {SPLIT_CONFIG[label]['val']}")
print(f"\n📁 Gespeichert: {len(train_set)} in '{TRAIN_FILE}', {len(val_set)} in '{VAL_FILE}'")
