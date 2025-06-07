from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from tqdm import tqdm

class SpectralCluster:
    def __init__(self, random_state=0):
        self.__random_state = random_state

    def fit_and_evaluate(self, df, columns, n_clusters_range, affinity="rbf"):
        """
        Fit Spectral Clustering models for a range of cluster numbers and evaluate metrics.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            columns (list): List of column names to be used in the model.
            n_clusters_range (range): Range of cluster numbers to fit the models for.
            affinity (str): Affinity type for Spectral Clustering ('rbf', 'nearest_neighbors', etc.).

        Returns:
            dict: Dictionary containing fitted models, evaluation metrics, and cluster labels.
        """
        data = df[columns].values  # Convert to numpy array for clustering

        models = []
        silhouettes = []
        silhouette_values_per_cluster = {}
        davies_bouldin_indices = []
        calinski_harabasz_indices = []

        for n in tqdm(n_clusters_range, desc="Fitting Spectral Clustering Models"):
            # Fit Spectral Clustering model
            spectral = SpectralClustering(
                n_clusters=n,
                affinity=affinity,
                random_state=self.__random_state,
                assign_labels="kmeans",
            )
            labels = spectral.fit_predict(data)
            models.append(spectral)

            # Calculate metrics
            if n > 1:  # Metrics only make sense for n > 1
                silhouette_avg = silhouette_score(data, labels)
                silhouettes.append(silhouette_avg)
                silhouette_values_per_cluster[n] = silhouette_samples(data, labels)
                davies_bouldin_indices.append(davies_bouldin_score(data, labels))
                calinski_harabasz_indices.append(calinski_harabasz_score(data, labels))
            else:
                silhouettes.append(None)
                silhouette_values_per_cluster[n] = None
                davies_bouldin_indices.append(None)
                calinski_harabasz_indices.append(None)

        return {
            "models": models,
            "silhouettes": silhouettes,
            "silhouette_values_per_cluster": silhouette_values_per_cluster,
            "davies_bouldin_indices": davies_bouldin_indices,
            "calinski_harabasz_indices": calinski_harabasz_indices,
        }
