from sklearn.cluster import KMeans
from src.utils.cluster.cluster_metrics import ClusterMetrics
from sklearn.metrics import silhouette_samples
from tqdm import tqdm
import pandas as pd


class KMeansCluster:
    def __init__(self, random_state=0):
        self.__random_state = random_state

    def fit_and_evaluate(self, df, columns, n_clusters_range):
        """
        Fit K-Means models for a range of cluster numbers and evaluate metrics.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            columns (list): List of column names to be used in the model.
            n_clusters_range (range): Range of cluster numbers to fit the models for.

        Returns:
            dict: Dictionary containing fitted models, evaluation metrics, and cluster labels.
        """
        data = df[columns].copy()

        models = []
        silhouettes = []
        silhouette_values_per_cluster = {}
        inertia_values = []
        davies_bouldin_indices = []
        calinski_harabasz_indices = []

        for n in tqdm(n_clusters_range, desc='Fitting K-Means Models'):
            # Fit K-Means model
            kmeans = KMeans(n_clusters=n, random_state=self.__random_state).fit(data)
            models.append(kmeans)

            # Predict cluster labels
            labels = kmeans.labels_

            # Calculate metrics
            metrics = ClusterMetrics.calculate_metrics(data, labels)

            silhouettes.append(metrics["silhouette_score"])
            silhouette_values = silhouette_samples(data, labels) if metrics["silhouette_score"] else None
            silhouette_values_per_cluster[n] = silhouette_values
            inertia_values.append(kmeans.inertia_)
            davies_bouldin_indices.append(metrics["davies_bouldin_index"])
            calinski_harabasz_indices.append(metrics["calinski_harabasz_index"])

        return {
            "models": models,
            "silhouettes": silhouettes,
            "silhouette_values_per_cluster": silhouette_values_per_cluster,
            "inertia_values": inertia_values,
            "davies_bouldin_indices": davies_bouldin_indices,
            "calinski_harabasz_indices": calinski_harabasz_indices
        }
        
        
        