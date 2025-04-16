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
    labels_spectral = np.load("../output/spectral_labels.npy")

    samenum_features = np.load("../data/r18_features/features_samenum.npy")
    samenum_labels_agglomerative = np.load("../output/agglomerative_samenum_labels.npy")
    samenum_labels_kmeans = np.load("../output/kmeans_samenum_labels.npy")
    samenum_labels_dbscan = np.load("../output/dbscan_samenum_labels.npy")
    samenum_labels_spectral = np.load("../output/spectral_samenum_labels.npy")
    
    # Calculate silhouette scores SAMENUM
    samenum_silhouette_agglomerative = silhouette_score(samenum_features, samenum_labels_agglomerative)
    samenum_silhouette_kmeans = silhouette_score(samenum_features, samenum_labels_kmeans)
    samenum_silhouette_dbscan = silhouette_score(samenum_features, samenum_labels_dbscan)
    samenum_silhouette_spectral = silhouette_score(samenum_features, samenum_labels_spectral)

    # Calculate silhouette scores
    silhouette_agglomerative = silhouette_score(features, labels_agglomerative)
    silhouette_kmeans = silhouette_score(features, labels_kmeans)
    silhouette_dbscan = silhouette_score(features, labels_dbscan)
    silhouette_spectral = silhouette_score(features, labels_spectral)

    # Calculate Davies-Bouldin index SAMENUM
    samenum_db_agglomerative = davies_bouldin_score(samenum_features, samenum_labels_agglomerative)
    samenum_db_kmeans = davies_bouldin_score(samenum_features, samenum_labels_kmeans)
    samenum_db_dbscan = davies_bouldin_score(samenum_features, samenum_labels_dbscan)
    samenum_db_spectral = davies_bouldin_score(samenum_features, samenum_labels_spectral)
    
    # Calculate Davies-Bouldin index
    db_agglomerative = davies_bouldin_score(features, labels_agglomerative)
    db_kmeans = davies_bouldin_score(features, labels_kmeans)
    db_dbscan = davies_bouldin_score(features, labels_dbscan)
    db_spectral = davies_bouldin_score(features, labels_spectral)

    # Calculate Calinski-Harabasz index SAMENUM
    samenum_ch_agglomerative = calinski_harabasz_score(samenum_features, samenum_labels_agglomerative)
    samenum_ch_kmeans = calinski_harabasz_score(samenum_features, samenum_labels_kmeans)
    samenum_ch_dbscan = calinski_harabasz_score(samenum_features, samenum_labels_dbscan)
    samenum_ch_spectral = calinski_harabasz_score(samenum_features, samenum_labels_spectral)
    
    # Calculate Calinski-Harabasz index
    ch_agglomerative = calinski_harabasz_score(features, labels_agglomerative)
    ch_kmeans = calinski_harabasz_score(features, labels_kmeans)
    ch_dbscan = calinski_harabasz_score(features, labels_dbscan)
    ch_spectral = calinski_harabasz_score(features, labels_spectral)

    # Print summary
    print("\nSummary of clustering metrics:")
    print(f"Agglomerative Clustering: Silhouette: {silhouette_agglomerative}, Davies-Bouldin: {db_agglomerative}, Calinski-Harabasz: {ch_agglomerative}")
    print(f"KMeans Clustering: Silhouette: {silhouette_kmeans}, Davies-Bouldin: {db_kmeans}, Calinski-Harabasz: {ch_kmeans}")
    print(f"DBSCAN Clustering: Silhouette: {silhouette_dbscan}, Davies-Bouldin: {db_dbscan}, Calinski-Harabasz: {ch_dbscan}")
    print(f"Spectral KNN Clustering: Silhouette: {silhouette_spectral}, Davies-Bouldin: {db_spectral}, Calinski-Harabasz: {ch_spectral}")
   
    print("\n\nSummary of clustering metrics SAMENUM:")
    print(f"Agglomerative Clustering: Silhouette: {samenum_silhouette_agglomerative}, Davies-Bouldin: {samenum_db_agglomerative}, Calinski-Harabasz: {samenum_ch_agglomerative}")
    print(f"KMeans Clustering: Silhouette: {samenum_silhouette_kmeans}, Davies-Bouldin: {samenum_db_kmeans}, Calinski-Harabasz: {samenum_ch_kmeans}")
    print(f"DBSCAN Clustering: Silhouette: {samenum_silhouette_dbscan}, Davies-Bouldin: {samenum_db_dbscan}, Calinski-Harabasz: {samenum_ch_dbscan}")
    print(f"Spectral KNN Clustering: Silhouette: {samenum_silhouette_spectral}, Davies-Bouldin: {samenum_db_spectral}, Calinski-Harabasz: {samenum_ch_spectral}")

if __name__ == "__main__":
    main()
    