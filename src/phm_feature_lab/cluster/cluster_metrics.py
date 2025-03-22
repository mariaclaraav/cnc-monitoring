from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import matplotlib.pyplot as plt

class ClusterMetrics:
    @staticmethod
    def calculate_metrics(data, labels):
        """
        Calculate clustering evaluation metrics.

        Parameters:
            data (pd.DataFrame or np.ndarray): Dataset used for clustering.
            labels (np.ndarray): Cluster labels for each data point.

        Returns:
            dict: Dictionary containing Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
        """
        metrics = {
            "silhouette_score": silhouette_score(data, labels) if len(set(labels)) > 1 else None,
            "davies_bouldin_index": davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None,
            "calinski_harabasz_index": calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
        }
        return metrics

    @staticmethod
    def plot_silhouette(n_clusters, silhouette_values, labels):
        """
        Plot the silhouette plot for the given number of clusters.

        Parameters:
            n_clusters (int): Number of clusters.
            silhouette_values (array): Array of silhouette values for each sample.
            labels (array): Cluster labels for each sample.
        """
        y_lower = 10
        plt.figure(figsize=(8, 6))

        for i in range(n_clusters):
            ith_cluster_silhouette_values = silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10  # 10 for spacing between silhouette plots

        plt.xlabel("Silhouette values")
        plt.ylabel("Cluster")
        plt.axvline(x=np.mean(silhouette_values), color="red", linestyle="--")
        plt.title(f"Silhouette plot for {n_clusters} clusters")
        plt.show()
    
    @staticmethod
    def plot_metrics(results, metrics, n_components_range=None, titles=None, ylabel=None):
        """
        Plot multiple metrics for clustering models.

        Parameters:
            results (dict): Dictionary containing metrics from clustering models.
            metrics (list): List of keys in the `results` dictionary to plot.
            n_components_range (range, optional): Range of the number of components. Defaults to range inferred from `results`.
            titles (list, optional): Titles for the subplots. Defaults to metric names.
            ylabel (list, optional): Y-axis labels for the subplots. Defaults to metric names.
        """
        n_components = n_components_range or range(2, len(results['models']) + 2)

        # Ensure titles and ylabels match metrics length
        titles = titles or metrics
        ylabel = ylabel or metrics

        # Number of metrics to plot
        num_metrics = len(metrics)

        # Determine layout (2 plots per row)
        n_rows = (num_metrics + 1) // 2  # Rows needed for 2 plots per row

        # Create subplots for each metric
        plt.figure(figsize=(16, 4 * n_rows))
        for i, metric in enumerate(metrics):
            plt.subplot(n_rows, 2, i + 1)
            plt.plot(n_components, results[metric], marker='o', label=metric)
            plt.xlabel('Number of Components')
            plt.ylabel(ylabel[i])
            plt.title(titles[i])
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()
        
    @staticmethod  
    def get_model(results, n_clusters):
        """
        Retrieve a clustering model trained with a specific number of clusters.

        Parameters:
            results (dict): Dictionary containing clustering results with a 'models' key.
            n_clusters (int): Desired number of clusters.
            cluster_start (int, optional): Starting number of clusters for indexing. Defaults to 2.

        Returns:
            object: Trained model with the specified number of clusters.

        Raises:
            ValueError: If no model is found for the given number of clusters.
        """
        # Adjust index based on the starting number of clusters
        cluster_start = 2
        index = n_clusters - cluster_start

        if 0 <= index < len(results['models']):
            return results['models'][index]
        else:
            raise ValueError(f"No model found for {n_clusters} clusters. Available range: "
                                f"{cluster_start} to {cluster_start + len(results['models']) - 1}.")

    @staticmethod
    def predict_clusters(model, data):
        """
        Predict cluster labels for the given data using a trained clustering model.

        Parameters:
            model (object): A trained clustering model with a `predict` method.
            data (pd.DataFrame or np.ndarray): Data for which to predict cluster labels.

        Returns:
            np.ndarray: Cluster labels for the input data.
        """
        if not hasattr(model, 'predict'):
            raise ValueError(f"The provided model does not have a 'predict' method.")

        return model.predict(data)