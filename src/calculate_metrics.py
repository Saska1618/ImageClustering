import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    
def main():
    """
    Main function to run the metrics.
    """
    # Read npy features
    features = np.load("../data/r18_features/features.npy")
    labels_agglomerative = np.load("../output/agglomerative_labels.npy")
    labels_kmeans = np.load("../output/kmeans_labels.npy")
    labels_dbscan = np.load("../output/dbscan_labels.npy")
    
    # Calculate silhouette scores
    silhouette_agglomerative = silhouette_score(features, labels_agglomerative)
    silhouette_kmeans = silhouette_score(features, labels_kmeans)
    silhouette_dbscan = silhouette_score(features, labels_dbscan)

    # Calculate Davies-Bouldin index
    db_agglomerative = davies_bouldin_score(features, labels_agglomerative)
    db_kmeans = davies_bouldin_score(features, labels_kmeans)
    db_dbscan = davies_bouldin_score(features, labels_dbscan)

    # Calculate Calinski-Harabasz index
    ch_agglomerative = calinski_harabasz_score(features, labels_agglomerative)
    ch_kmeans = calinski_harabasz_score(features, labels_kmeans)
    ch_dbscan = calinski_harabasz_score(features, labels_dbscan)

    # Print summary
    print("\nSummary of clustering metrics:")
    print(f"Agglomerative Clustering: Silhouette: {silhouette_agglomerative}, Davies-Bouldin: {db_agglomerative}, Calinski-Harabasz: {ch_agglomerative}")
    print(f"KMeans Clustering: Silhouette: {silhouette_kmeans}, Davies-Bouldin: {db_kmeans}, Calinski-Harabasz: {ch_kmeans}")
    print(f"DBSCAN Clustering: Silhouette: {silhouette_dbscan}, Davies-Bouldin: {db_dbscan}, Calinski-Harabasz: {ch_dbscan}")
   

if __name__ == "__main__":
    main()
    