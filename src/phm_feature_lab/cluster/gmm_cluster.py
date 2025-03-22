from phm_feature_lab.utils.cluster.cluster_metrics import ClusterMetrics
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

class GMMCluster:
    def __init__(self, covariance_type='diag', random_state=0):
        self.__covariance_type = covariance_type
        self.__random_state = random_state

    def fit_and_evaluate(self, df, columns, n_components_range):
        """
        Fit Gaussian Mixture Models for a range of component numbers and evaluate metrics.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            columns (list): List of column names to be used in the model.
            n_components_range (range): Range of component numbers to fit the models for.

        Returns:
            dict: Dictionary containing fitted models, evaluation metrics, and silhouette values.
        """
        data = df[columns].copy()

        models = []
        silhouettes = []
        silhouette_values_per_component = {}
        bics = []
        davies_bouldin_indices = []
        calinski_harabasz_indices = []

        for n in tqdm(n_components_range, desc='Fitting GMM Models'):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.__covariance_type,
                random_state=self.__random_state
            ).fit(data)
            models.append(gmm)

            labels = gmm.predict(data)
            metrics = ClusterMetrics.calculate_metrics(data, labels)

            silhouettes.append(metrics["silhouette_score"])
            silhouette_values = silhouette_samples(data, labels) if metrics["silhouette_score"] else None
            silhouette_values_per_component[n] = silhouette_values
            davies_bouldin_indices.append(metrics["davies_bouldin_index"])
            calinski_harabasz_indices.append(metrics["calinski_harabasz_index"])

            bics.append(gmm.bic(data))

        return {
            "models": models,
            "silhouettes": silhouettes,
            "silhouette_values_per_component": silhouette_values_per_component,
            "bics": bics,
            "davies_bouldin_indices": davies_bouldin_indices,
            "calinski_harabasz_indices": calinski_harabasz_indices
        }
