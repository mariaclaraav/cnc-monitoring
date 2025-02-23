from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
#from sklearn_extra.cluster import KMedoids
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from typing import List, Optional, Dict, Any, Tuple, Union

# Configuração básica do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    
    
def fit_gmm_evaluate(df, columns, n_components_range, random_state=0, covariance_type='diag'):
    """
    Fit Gaussian Mixture Models for a range of component numbers and evaluate using several metrics.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        columns (list): List of column names to be used in the model.
        n_components_range (range): Range of component numbers to fit the models for.
        random_state (int): Random state for reproducibility of the models.
        covariance_type (str): Type of covariance parameters to use in the GMM.
        
    Returns:
        dict: Dictionary containing fitted models, evaluation metrics, and silhouette values for each sample.
    """
    # Prepare the data
    data = df[columns].copy()
    
    # Storage for models and metrics
    models = []
    silhouettes = []
    silhouette_values_per_component = {}
    bics = []
    davies_bouldin_indices = []
    calinski_harabasz_indices = []
    
    # Fit models and compute metrics
    for n in tqdm(n_components_range, desc='Fitting Models'):
        gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=random_state,
                              verbose=1, verbose_interval=10).fit(data)
        models.append(gmm)
        
        # Predict the labels
        labels = gmm.predict(data)
        
        # Calculate metrics if there is more than one cluster
        if n > 1:
            silhouette_avg = silhouette_score(data, labels)
            silhouettes.append(silhouette_avg)
            silhouette_values = silhouette_samples(data, labels)
            silhouette_values_per_component[n] = silhouette_values
            davies_bouldin_indices.append(davies_bouldin_score(data, labels))
            calinski_harabasz_indices.append(calinski_harabasz_score(data, labels))
        else:
            silhouettes.append(None)
            silhouette_values_per_component[n] = None
            davies_bouldin_indices.append(None)
            calinski_harabasz_indices.append(None)
        
        # Calculate BIC
        bics.append(gmm.bic(data))
    
    return {
        "models": models,
        "silhouettes": silhouettes,
        "silhouette_values_per_component": silhouette_values_per_component,
        "bics": bics,
        "davies_bouldin_indices": davies_bouldin_indices,
        "calinski_harabasz_indices": calinski_harabasz_indices
    }


def fit_kmeans_evaluate(data , n_clusters, random_state=0):
    """
    Fit a K-Means model for a specific number of clusters and evaluate using several metrics.
    """
    
    # Fit the K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
    
    # Predict the labels
    labels = kmeans.labels_
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(data, labels)
    silhouette_values = silhouette_samples(data, labels)
    davies_bouldin_idx = davies_bouldin_score(data, labels)
    calinski_harabasz_idx = calinski_harabasz_score(data, labels)
    
    return {
        "model": kmeans,
        "inertia": inertia,
        "silhouette_score": silhouette_avg,
        "davies_bouldin_index": davies_bouldin_idx,
        "calinski_harabasz_index": calinski_harabasz_idx,
        "silhouette_values": silhouette_values,
        "labels": labels
    }
    
    
# def fit_kmedoids_evaluate(df, columns, n_clusters_range, random_state=0, metric='euclidean'):
#     """
#     Fit K-Medoids models for a range of cluster numbers and evaluate using several metrics.
    
#     Parameters:
#         n_clusters_range (range): Range of cluster numbers to fit the models for.
#         metric (str): The metric to use when calculating distance between instances in a feature array.
#     """
#     # Prepare the data
#     data = df[columns].copy()
    
#     # Storage for models and metrics
#     models = []
#     silhouettes = []
#     silhouette_values_per_cluster = {}
#     davies_bouldin_indices = []
#     calinski_harabasz_indices = []
    
#     # Fit models and compute metrics
#     for n_clusters in tqdm(n_clusters_range, desc='Fitting K-Medoids Models'):
#         kmedoids = KMedoids(n_clusters=n_clusters, random_state=random_state, metric=metric).fit(data)
#         models.append(kmedoids)
        
#         # Predict the labels
#         labels = kmedoids.labels_
        
#         # Calculate metrics
#         if n_clusters > 1:
#             silhouette_avg = silhouette_score(data, labels)
#             silhouettes.append(silhouette_avg)
#             silhouette_values = silhouette_samples(data, labels)
#             silhouette_values_per_cluster[n_clusters] = silhouette_values
#             davies_bouldin_indices.append(davies_bouldin_score(data, labels))
#             calinski_harabasz_indices.append(calinski_harabasz_score(data, labels))
#         else:
#             silhouettes.append(None)
#             silhouette_values_per_cluster[n_clusters] = None
#             davies_bouldin_indices.append(None)
#             calinski_harabasz_indices.append(None)
    
#     return {
#         "models": models,
#         "silhouettes": silhouettes,
#         "silhouette_values_per_cluster": silhouette_values_per_cluster,
#         "davies_bouldin_indices": davies_bouldin_indices,
#         "calinski_harabasz_indices": calinski_harabasz_indices
#     }
    

# def hierarchial_clustering_metrics(df, methods, criteria, range_n_clusters):
#     '''' hierarchial_clustering_metrics

#     Calculate clustering evaluation metrics for hierarchical clustering using different methods, criteria, and a range of cluster sizes.

#     Args
#     - df: DataFrame containing the data.
#     - methods: List of methods for hierarchical clustering.
#     - criteria: List of criteria for clustering evaluation.
#     - range_n_clusters: Range of cluster sizes to evaluate.

#     Returns
#     Tuple of dictionaries containing silhouette scores, Davies-Bouldin indices, and Calinski-Harabasz indices for each method and criterion.'''

#     # Dictionaries to store the scores
#     silhouette_scores = {(method, criterion): [] for method in methods for criterion in criteria}
#     davies_bouldin_indices = {(method, criterion): [] for method in methods for criterion in criteria}
#     calinski_harabasz_indices = {(method, criterion): [] for method in methods for criterion in criteria}

#     # Perform linkage for each method once
#     results = {method: linkage(df, method=method) for method in methods}

#     def add_scores(method, criterion, n_clusters, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg):
#         silhouette_scores[(method, criterion)].append(silhouette_avg)
#         davies_bouldin_indices[(method, criterion)].append(davies_bouldin_avg)
#         calinski_harabasz_indices[(method, criterion)].append(calinski_harabasz_avg)

#     for method in methods:
#         Z = results[method]
#         for criterion in criteria:
#             for n_clusters in range_n_clusters:
#                 try:
#                     cluster_labels = fcluster(Z, n_clusters, criterion=criterion)
#                 except ValueError as e:
#                     print(f'Error for method {method} with criterion {criterion} and n_clusters {n_clusters}: {e}')
#                     add_scores(method, criterion, n_clusters, None, None, None)
#                     continue

#                 n_labels = len(np.unique(cluster_labels))
#                 if n_labels < 2 or n_labels > len(df) - 1:
#                     add_scores(method, criterion, n_clusters, None, None, None)
#                     continue

#                 silhouette_avg = silhouette_score(df, cluster_labels)
#                 davies_bouldin_avg = davies_bouldin_score(df, cluster_labels)
#                 calinski_harabasz_avg = calinski_harabasz_score(df, cluster_labels)

#                 add_scores(method, criterion, n_clusters, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg)
#                 print(f'Method: {method}, Criterion: {criterion}, Num Clusters: {n_clusters}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {davies_bouldin_avg}, Calinski-Harabasz Index: {calinski_harabasz_avg}')
    
#     return silhouette_scores, davies_bouldin_indices, calinski_harabasz_indices

def hierarchical_clustering_metrics(df: pd.DataFrame, 
                                    methods: List[str], 
                                    criteria: List[str], 
                                    range_n_clusters: Optional[range] = None, 
                                    thresholds: Optional[Union[float, List[float]]] = None, 
                                    depths: Optional[Union[int, List[int]]] = 2) -> Dict[str, Dict[Tuple[str, str], List[float]]]:
    '''
    Calculate clustering evaluation metrics for hierarchical clustering using different methods, criteria, 
    and a range of cluster sizes or thresholds.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        methods (List[str]): List of methods for hierarchical clustering.
        criteria (List[str]): List of criteria for clustering evaluation.
        range_n_clusters (Optional[range]): Range of cluster sizes to evaluate (used with 'maxclust').
        thresholds (Optional[Union[float, List[float]]]): Single or range of threshold values for 'distance', 'inconsistent', or 'threshold' criteria.
        depths (Optional[Union[int, List[int]]]): Single or range of depths used for the 'inconsistent' criterion.

    Returns:
        Dict[str, Dict[Tuple[str, str], List[float]]]: Dictionary containing silhouette scores, Davies-Bouldin indices,
        Calinski-Harabasz indices, and silhouette values for each sample.
    '''

    # Normalize thresholds and depths to lists if they are not already
    if thresholds is not None and not isinstance(thresholds, list):
        thresholds = [thresholds]
    if not isinstance(depths, list):
        depths = [depths]

    # Dictionaries to store the scores
    silhouette_scores = {(method, criterion): [] for method in methods for criterion in criteria}
    davies_bouldin_indices = {(method, criterion): [] for method in methods for criterion in criteria}
    calinski_harabasz_indices = {(method, criterion): [] for method in methods for criterion in criteria}
    silhouette_values_per_cluster = {(method, criterion): {} for method in methods for criterion in criteria}

    # Perform linkage for each method once
    results = {method: linkage(df, method=method) for method in methods}

    def add_scores(method: str, 
                   criterion: str, 
                   n_clusters: int, 
                   silhouette_avg: float, 
                   davies_bouldin_avg: float, 
                   calinski_harabasz_avg: float, 
                   silhouette_values: np.ndarray) -> None:
        '''Add computed scores to their respective dictionaries.'''
        silhouette_scores[(method, criterion)].append(silhouette_avg)
        davies_bouldin_indices[(method, criterion)].append(davies_bouldin_avg)
        calinski_harabasz_indices[(method, criterion)].append(calinski_harabasz_avg)
        silhouette_values_per_cluster[(method, criterion)][n_clusters] = silhouette_values

    for method in methods:
        Z = results[method]
        for criterion in criteria:
            if criterion == 'maxclust':
                if range_n_clusters is None:
                    raise ValueError("range_n_clusters must be provided for 'maxclust' criterion")
                for n_clusters in range_n_clusters:
                    cluster_labels = fcluster(Z, t=n_clusters, criterion=criterion)
                    actual_n_clusters = len(np.unique(cluster_labels))
                    silhouette_avg = silhouette_score(df, cluster_labels)
                    silhouette_values = silhouette_samples(df, cluster_labels)
                    davies_bouldin_avg = davies_bouldin_score(df, cluster_labels)
                    calinski_harabasz_avg = calinski_harabasz_score(df, cluster_labels)
                    add_scores(method, criterion, actual_n_clusters, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg, silhouette_values)
                    logging.info(f'Method: {method}, Criterion: {criterion}, Requested Clusters: {n_clusters}, Actual Clusters: {actual_n_clusters}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {davies_bouldin_avg}, Calinski-Harabasz Index: {calinski_harabasz_avg}')

            elif criterion in ['inconsistent', 'distance', 'threshold']:
                if thresholds is None:
                    raise ValueError(f"Threshold(s) must be provided for '{criterion}' criterion")
                for t in thresholds:
                    for depth in depths:
                        cluster_labels = fcluster(Z, t=t, criterion=criterion, depth=depth)
                        actual_n_clusters = len(np.unique(cluster_labels))
                        silhouette_avg = silhouette_score(df, cluster_labels)
                        silhouette_values = silhouette_samples(df, cluster_labels)
                        davies_bouldin_avg = davies_bouldin_score(df, cluster_labels)
                        calinski_harabasz_avg = calinski_harabasz_score(df, cluster_labels)
                        add_scores(method, criterion, actual_n_clusters, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg, silhouette_values)
                        logging.info(f'Method: {method}, Criterion: {criterion}, Threshold: {t}, Depth: {depth}, Actual Clusters: {actual_n_clusters}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {davies_bouldin_avg}, Calinski-Harabasz Index: {calinski_harabasz_avg}')
            
            else:
                raise ValueError(f"Unsupported criterion: {criterion}")
    
    return {
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_indices": davies_bouldin_indices,
        "calinski_harabasz_indices": calinski_harabasz_indices,
        "silhouette_values_per_cluster": silhouette_values_per_cluster
    }



def hdbscan_clustering_metrics(df, min_cluster_sizes, min_samples_list):
    '''hdbscan_clustering_metrics
    
    Calculate clustering evaluation metrics for HDBSCAN using different min_cluster_size and min_samples parameters.

    Args:
    - df: DataFrame containing the data.
    - min_cluster_sizes: List of min_cluster_size values to evaluate.
    - min_samples_list: List of min_samples values to evaluate.

    Returns:
    Tuple of dictionaries containing silhouette scores, Davies-Bouldin indices, and Calinski-Harabasz indices for each combination of parameters.
    '''
    
    # Dictionaries to store the scores
    silhouette_scores = {}
    davies_bouldin_indices = {}
    calinski_harabasz_indices = {}
    
    def add_scores(min_cluster_size, min_samples, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg):
        silhouette_scores[(min_cluster_size, min_samples)] = silhouette_avg
        davies_bouldin_indices[(min_cluster_size, min_samples)] = davies_bouldin_avg
        calinski_harabasz_indices[(min_cluster_size, min_samples)] = calinski_harabasz_avg
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                cluster_labels = clusterer.fit_predict(df)
            except ValueError as e:
                print(f'Error for min_cluster_size {min_cluster_size} and min_samples {min_samples}: {e}')
                add_scores(min_cluster_size, min_samples, None, None, None)
                continue

            n_labels = len(np.unique(cluster_labels))
            if n_labels < 2 or n_labels > len(df) - 1:
                add_scores(min_cluster_size, min_samples, None, None, None)
                continue

            silhouette_avg = silhouette_score(df, cluster_labels)
            davies_bouldin_avg = davies_bouldin_score(df, cluster_labels)
            calinski_harabasz_avg = calinski_harabasz_score(df, cluster_labels)

            add_scores(min_cluster_size, min_samples, silhouette_avg, davies_bouldin_avg, calinski_harabasz_avg)
            print(f'Min Cluster Size: {min_cluster_size}, Min Samples: {min_samples}, Silhouette Score: {silhouette_avg}, Davies-Bouldin Index: {davies_bouldin_avg}, Calinski-Harabasz Index: {calinski_harabasz_avg}')
    
    return silhouette_scores, davies_bouldin_indices, calinski_harabasz_indices
