import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import geopandas as gpd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
def clustering(
        dataframe: pd.DataFrame,
        columns_to_cluster: list,
        nb_cluster_min: int,
        nb_cluster_max: int,
        title: str,
        step: int = 1,
        show_silouette_scores: bool = False
) -> int:
    """
        Function that determines the optimal number of clusters based on silhouette score
        :param dataframe: dataframe containing the data to cluster
        :param columns_to_cluster: columns to cluster
        :param nb_cluster_min: minimum number of clusters to test
        :param nb_cluster_max: maximal number of clusters to test
        :param step: step between the minimal and the maximal
        :param show_silhouette_scores: show a plot of the evolution of the silhouette score for each number of clusters
        :return: best number of clusters based on the silhouette score
    """

    data_to_cluster = dataframe[columns_to_cluster].values
    silhouette_scores = []

    for n_clusters in range(nb_cluster_min, nb_cluster_max, step):
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(data_to_cluster)
        silhouette = silhouette_score(data_to_cluster, cluster_labels)
        silhouette_scores.append(silhouette)

    if show_silouette_scores:
        plt.figure(figsize=(12, 8))
        plt.plot(range(nb_cluster_min, nb_cluster_max), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters', fontsize=15)
        plt.xticks(range(nb_cluster_min, nb_cluster_max))
        plt.ylabel('Silhouette score', fontsize=15)
        plt.title(title, fontsize=15)
        plt.show()

    list_nb_cluster = list(range(nb_cluster_min, nb_cluster_max + 1))
    index_best_nb_cluster = silhouette_scores.index(max(silhouette_scores))

    return list_nb_cluster[index_best_nb_cluster]


def visualize_clusters_map(
        cluster_labels: np.ndarray,
        title: str
):
    """
    Visualize the clusters of UHF42 on the map
    :param cluster_labels: the clusters to be predicted
    :param title: title of the visualization
    """
    # reading file containing the geodata used for visualization
    file = os.path.join('UHF42', 'UHF42.shp')
    geo_data = gpd.read_file(file)
    geo_data.drop(0, inplace=True)

    # adding the cluster number
    geo_data['Cluster'] = cluster_labels

    # visualizing the clusters on map
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    geo_data.plot(column='Cluster', legend=True, edgecolor='black', ax=ax, cmap='viridis',
                  categorical=True,
                  legend_kwds={'fontsize': 12, 'title': "Cluster Number", 'title_fontsize': 12, 'loc': 'upper left'
                               })
    plt.title(title, fontsize=20)

    # adding the numbers of the UHF42 o the map
    for idx, row in geo_data.iterrows():
        ax.annotate(text=row['id'], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=12, color='white')
    ax.set_axis_off()


def feature_importance(
        dataframe: pd.DataFrame,
        predictors: list,
        target: np.ndarray,
        title: str
):
    """
    Function to visualize the feature importance in predicting the cluster label
    :param dataframe: dataframe containing the features
    :param predictors: predictors that will be used to predict the cluster label
    :param target: cluster label that will be predicted
    :param title: title of the visualization
    """
    # using extra tree classifier to predict the cluster labels based on predictor passed as parameters
    x = dataframe[predictors]
    y = target
    model = ExtraTreesClassifier(random_state=2)
    model.fit(x, y)

    # Get the feature importances
    importance = model.feature_importances_

    # Sort the features according to their importance
    indices = np.argsort(importance)

    # Rearrange the features names so they match the sorted feature importances
    features = x.columns.values[indices]
    importances = importance[indices] * 100

    # Plot the feature importances
    plt.figure(figsize=(9, 9))
    plt.barh(features, importances, orientation='horizontal', color="#008bfb")
    for index, value in enumerate(importances):
        plt.text(value, index, f'{value:.2f}%', va='center', fontweight='bold', ha='left')
    plt.title(title)
    sns.despine()
    plt.show()