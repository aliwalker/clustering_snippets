import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

def gen_sample():
    '''
    Generate sample from sklearn.
    '''
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return X, y

def elbow_method(X):
    '''
    Train multiple models using a different number of
    clusters and plot the graph of [Within Cluster Sum of Squares].
    This is for selecting the number of clusters heuristically. We'll choose
    the number of clusters where the change in WCSS begins to level off.

    WSS = sum(xi - ci)^2
    WCSS = sum(WSS)
    '''
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def main():
    X, y = gen_sample()
    elbow_method(X)

if __name__ == '__main__':
    main()