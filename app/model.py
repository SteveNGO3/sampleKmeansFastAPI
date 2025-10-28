# model.py
from sklearn.cluster import KMeans
import numpy as np

# Exemple de données d'entraînement
X_train = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
model = KMeans(n_clusters=2)
model.fit(X_train)