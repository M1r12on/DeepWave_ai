import json
import numpy as np
import soundfile as sf
import os
from tqdm import tqdm

LABEL_TO_INDEX = {
    "Lead": 0,
    "Pluck": 1,
    "Bass": 2,
    "Pad": 3
}

def extract_cpp_features(filepath):
    try:
        y, sr = sf.read(filepath)

        # Mono erzwingen
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Stille oder fehlerhafte Datei überspringen
        if len(y) == 0 or np.max(np.abs(y)) == 0:
            print(f"Nur Stille oder ungültige Datei: {filepath}")
            return None

        # Feature 1: Dauer
        duration = len(y) / sr

        # Feature 2: Energie
        energy = np.sum(y ** 2)

        # Feature 3: RMS
        rms = np.sqrt(energy / len(y))

        # Feature 4: ZCR
        zero_crossings = np.where(np.diff(np.signbit(y)))[0]
        zcr = len(zero_crossings) / len(y)

        # Feature 5: Crest-Faktor
        crest = np.max(np.abs(y)) / rms if rms > 0 else 0.0

        # === FFT-basierte Features ===
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1.0/sr)

        spectral_sum = np.sum(spectrum)

        # Feature 6: Spectral Centroid
        centroid = np.sum(freqs * spectrum) / spectral_sum if spectral_sum > 0 else 0.0

        # Feature 7: Spectral Bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0

        # Feature 8: Spectral Roll-off (z. B. 85 %)
        rolloff_threshold = 0.85 * spectral_sum
        cumulative_sum = np.cumsum(spectrum)
        rolloff = freqs[np.searchsorted(cumulative_sum, rolloff_threshold)] if spectral_sum > 0 else 0.0

        return [duration, energy, rms, zcr, crest, centroid, bandwidth, rolloff]

    except Exception as e:
        print(f"Fehler bei {filepath}: {e}")
        return None


def process_data(json_input, feature_output, label_output):
    with open(json_input, "r") as f:
        data = json.load(f)

    features_list = []
    labels_list = []

    for entry in tqdm(data, desc=f"🔍 Verarbeite {json_input}"):
        path = entry["file"]
        label = entry["label"]

        if not os.path.exists(path):
            print(f"Datei fehlt: {path}")
            continue

        features = extract_cpp_features(path)
        if features is None:
            continue

        features_list.append(features)
        labels_list.append(LABEL_TO_INDEX[label])

    os.makedirs("data", exist_ok=True)
    np.save(feature_output, np.array(features_list, dtype=np.float32))
    np.save(label_output, np.array(labels_list, dtype=np.int64))

    print(f"Gespeichert: {len(features_list)} Features → {feature_output}")
    print(f"Gespeichert: {len(labels_list)} Labels → {label_output}")


# 🔁 Aufruf
process_data("train_data.json", "data/train_features.npy", "data/train_labels.npy")
process_data("val_data.json", "data/val_features.npy", "data/val_labels.npy")
