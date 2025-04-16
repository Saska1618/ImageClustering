# Feature Clustering & Evaluation

This project applies clustering algorithms to high-dimensional feature vectors (e.g., extracted from images), visualizes the results using dimensionality reduction, and evaluates cluster quality using multiple metrics.

---

## 🧠 Clustering Algorithms

### 1. **KMeans**

- Centroid-based clustering.
- Assumes spherical clusters.
- `n_clusters = 4`

### 2. **DBSCAN**

- Density-based clustering.
- Detects arbitrary-shaped clusters and outliers.
- `eps = 20`, `min_samples = 3`

### 3. **Agglomerative Clustering**

- Hierarchical, bottom-up approach.
- Good for varied-sized and shaped clusters.
- `n_clusters = 4`

### 4. **Spectral Clustering**

- Graph-based clustering using similarity matrix.
- Suitable for non-convex clusters.
- `n_clusters = 4`, `affinity = 'nearest_neighbors'`, `n_neighbors = 89`

---

## 📉 Dimensionality Reduction

Before visualization:

- Reduce to 50D with **PCA**
- Then to 2D with **t-SNE**
- Cluster and real-label scatter plots are saved as `.png`

---

## 📊 Evaluation Metrics

Each algorithm is evaluated using:

### ✔️ Silhouette Score

- Measures cluster cohesion and separation.
- Range: -1 (bad) to 1 (good)

### ✔️ Davies-Bouldin Index

- Lower is better.
- Measures average similarity between clusters.

### ✔️ Calinski-Harabasz Index

- Higher is better.
- Ratio of between-cluster to within-cluster dispersion.

Metrics are computed for:

- Full feature set
- Balanced feature subset (`features_samenum.npy`)

---

## 📁 Inputs & Outputs

**Inputs:**

- `../data/r18_features/features.npy`
- `../data/r18_features/labels.npy`
- `../data/r18_features/features_samenum.npy`

**Outputs:**
- Predicted cluster labels in `../output/`
- Evaluation metrics printed in console
- Cluster visualizations saved as PNGs

---

## ▶️ Run Clustering

```bash
python clustering_script.py --clustering [kmeans|dbscan|agglomerative|spectral]
