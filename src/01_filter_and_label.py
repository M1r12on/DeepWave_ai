# Datei: 01_filter_and_label.py
import os
import json
import pandas as pd

CSV_PATH = "../metadata/UrbanSound8K.csv"
AUDIO_ROOT = "../audio"
OUTPUT_PATH = "filtered_data.json"

# Zielklassen → Synth-Zuordnung
CLASS_TO_SYNTH = {
    "siren": "Lead",
    "dog_bark": "Pluck",
    "engine_idling": "Bass",
    "street_music": "Pad"
}

# Metadaten laden
meta = pd.read_csv(CSV_PATH)

# Nur gewünschte Klassen
selected = meta[meta["class"].isin(CLASS_TO_SYNTH.keys())]

# Dateien prüfen & umbenennen
valid_data = []
class_counts = {c: 0 for c in CLASS_TO_SYNTH.keys()}

for _, row in selected.iterrows():
    fold = f"fold{row['fold']}"
    filename = row["slice_file_name"]
    label = row["class"]
    path = os.path.join(AUDIO_ROOT, fold, filename)

    if os.path.exists(path):
        valid_data.append({
            "file": path,
            "label": CLASS_TO_SYNTH[label]
        })
        class_counts[label] += 1

# JSON speichern
with open(OUTPUT_PATH, "w") as f:
    json.dump(valid_data, f, indent=2)

# Übersicht ausgeben
print("🎧 Gültige Audiodateien pro Klasse:")
for cls, count in class_counts.items():
    print(f"{cls:<15} → {count:>4} Dateien")
print(f"\n✅ Gesamt: {len(valid_data)} Einträge gespeichert in {OUTPUT_PATH}")
