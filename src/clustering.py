import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def plot_clusters(features_2d, labels, title):
    '''
    Plot the clusters in 2D space.
    Args:
        features_2d (numpy.ndarray): 2D array of features.
        labels (numpy.ndarray): Cluster labels for each point.
        title (str): Title of the plot.
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(title)
    plt.savefig(f"../output/{title}.png")

def main():
    '''
    Main function to run the clustering.
    '''
    # Parse arguments
    parser = argparse.ArgumentParser(description='Clustering of features')

    # Add arguments
    parser.add_argument('--clustering', type=str, default='kmeans', help='Clustering algorithm to use (kmeans, dbscan, agglomerative)')

    # Validate arguments
    args = parser.parse_args()
    if args.clustering not in ['kmeans', 'dbscan', 'agglomerative', 'spectral']:
        raise ValueError("Invalid clustering algorithm. Choose from 'kmeans', 'dbscan', 'agglomerative', 'spectral'")

    # Read npy feautures
    features = np.load("../data/r18_features/features_samenum.npy")
    real_labels = np.load("../data/r18_features/labels_samenum.npy")
    num_clusters = 4

    # Use TSNE to reduce the dimensionality of the features
    # Reduce first to 50 dimensions using PCA
    # and then to 2 dimensions using TSNE
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    # Now reduce to 2 dimensions using TSNE
    # Note: TSNE is computationally expensive, so we use PCA first 
    # to reduce the dimensionality before applying TSNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_pca)

    kmean_model = KMeans(n_clusters=num_clusters, random_state=42)
    dbscan_model = DBSCAN(eps=20, min_samples=3)
    agglomerative_model = AgglomerativeClustering(n_clusters=num_clusters)
    spectral_model = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', n_neighbors=89, random_state=42)


    # Fit the model
    if args.clustering == 'kmeans':
        labels = kmean_model.fit_predict(features)
    elif args.clustering == 'dbscan':
        labels = dbscan_model.fit_predict(features)
    elif args.clustering == 'agglomerative':
        labels = agglomerative_model.fit_predict(features)
    elif args.clustering == 'spectral':
        labels = spectral_model.fit_predict(features)
    else:
        raise ValueError("Invalid clustering algorithm. Choose from 'kmeans', 'dbscan', 'agglomerative'")

    # Plot clusters
    plot_clusters(features_2d, labels, f"samenum {args.clustering} Clustering")
    # Plot real labels
    plot_clusters(features_2d, real_labels, "Real Labels")
    
    # Save the labels
    np.save(f"../output/{args.clustering}_samenum_labels.npy", labels)
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(features, labels)
    print(f"Silhouette Score for {args.clustering}: {silhouette_avg}")

if __name__ == "__main__":
    main()
    