import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
import os

# === Label-Mapping ===
INDEX_TO_LABEL = ["Lead", "Pluck", "Bass", "Pad"]

# === Feature-Extraktion ===
def extract_cpp_features(filepath):
    try:
        y, sr = sf.read(filepath)

        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if len(y) == 0 or np.max(np.abs(y)) == 0:
            print("⚠️ Ungültige Datei.")
            return None

        duration = len(y) / sr
        energy = np.sum(y ** 2)
        rms = np.sqrt(energy / len(y))
        zero_crossings = np.where(np.diff(np.signbit(y)))[0]
        zcr = len(zero_crossings) / len(y)
        crest = np.max(np.abs(y)) / rms if rms > 0 else 0.0

        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1.0/sr)
        spectral_sum = np.sum(spectrum)
        centroid = np.sum(freqs * spectrum) / spectral_sum if spectral_sum > 0 else 0.0
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0
        rolloff_threshold = 0.85 * spectral_sum
        cumulative_sum = np.cumsum(spectrum)
        rolloff = freqs[np.searchsorted(cumulative_sum, rolloff_threshold)] if spectral_sum > 0 else 0.0

        return [duration, energy, rms, zcr, crest, centroid, bandwidth, rolloff]

    except Exception as e:
        print(f"Fehler beim Lesen: {e}")
        return None

# === 1. Datei aus val_data.json laden ===
with open("val_data.json", "r") as f:
    val_data = json.load(f)

if len(val_data) == 0:
    raise ValueError("Keine Einträge in val_data.json gefunden.")

file_path = val_data[0]["file"]
print(f"Analysiere Datei: {file_path}")

features = extract_cpp_features(file_path)
if features is None:
    exit()

features = np.array(features, dtype=np.float32).reshape(1, -1)

# === Normierung mit Trainingsstatistiken ===
train_data = np.load("data/train_features.npy")
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
std[std == 0] = 1e-8
features_norm = (features - mean) / std

# === ONNX-Modell laden und Vorhersage durchführen ===
session = ort.InferenceSession("synth_classifier_cpp.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

output = session.run([output_name], {input_name: features_norm.astype(np.float32)})
predicted_class = int(np.argmax(output[0], axis=1)[0])

# === Ergebnis ausgeben ===
feature_names = ["Dauer", "Energie", "RMS", "ZCR", "Crest", "Centroid", "Bandwidth", "Roll-off"]

print("\nExtrahierte Merkmale:")
for name, value in zip(feature_names, features[0]):
    print(f" - {name:<10}: {value:.4f}")

print("\n🧮 Normalisierte Merkmale:")
for name, value in zip(feature_names, features_norm[0]):
    print(f" - {name:<10}: {value:.4f}")

print(f"\nVorhergesagte Klasse: {INDEX_TO_LABEL[predicted_class]}")
