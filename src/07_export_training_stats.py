import numpy as np

# Trainingsdaten laden
X_train = np.load("data/train_features.npy")

# Mittelwert und Standardabweichung berechnen
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1e-8  # gegen Division durch 0

# C++-kompatiblen Output erzeugen
print("float mean[] = {", ', '.join(f"{m:.6f}f" for m in mean), "};")
print("float std[] = {", ', '.join(f"{s:.6f}f" for s in std), "};")

# Optional: auch als .npy speichern
np.save("cpp_mean.npy", mean)
np.save("cpp_std.npy", std)
