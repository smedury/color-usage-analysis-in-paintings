import skimage
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from src.commons import utils
from src.preprocessing import preprocessing
from src.visualization import visualization
import numpy as np
import pandas as pd
from src.constants import *
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from pathlib import Path

def cluster_colors_rgb(pixel_matrix, n_clusters=10, filename='tmp.jpg'):
    #   Scale RGB colors
    pixel_matrix = preprocessing.preprocess_pixel_matrix(pixel_matrix)
    #   Create clust, fit and rescale colors
    clust = KMeans(n_clusters=n_clusters, verbose=3, random_state=999)
    clust.fit(pixel_matrix)

    cluster_centers = clust.cluster_centers_
    cluster_centers_rgb = cluster_centers * 255
    #cluster_centers_hsv = matplotlib.colors.rgb_to_hsv(cluster_centers)
    cluster_centers_hsv = skimage.color.rgb2hsv(cluster_centers)

    clusters_centers_df = pd.DataFrame()

    #   Save HSV
    clusters_centers_df['H'] = cluster_centers_hsv[:, 0]
    clusters_centers_df['S'] = cluster_centers_hsv[:, 1]
    clusters_centers_df['V'] = cluster_centers_hsv[:, 2]

    #   Save RGB
    clusters_centers_df['R'] = cluster_centers_rgb[:, 0]
    clusters_centers_df['G'] = cluster_centers_rgb[:, 1]
    clusters_centers_df['B'] = cluster_centers_rgb[:, 2]

    clusters_centers_df['R'] = clusters_centers_df['R'].astype(np.int64)
    clusters_centers_df['G'] = clusters_centers_df['G'].astype(np.int64)
    clusters_centers_df['B'] = clusters_centers_df['B'].astype(np.int64)

    df = pd.DataFrame()
    df['points'] = clust.labels_
    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']
    visualization.draw_colors(df, clusters_centers_df, filename)

    color_scheme = pd.DataFrame(clusters_centers_df, columns=['R', 'G', 'B', 'H', 'S', 'V'])
    color_scheme.to_csv('{}/color_scheme.csv'.format(OUTPUT_FOLDER))

    return clust

def generate_color_scheme_images(pixel_matrix, clust, filename):
    if pixel_matrix is not None:
        #   Scale RGB colors
        pixel_matrix = preprocessing.preprocess_pixel_matrix(pixel_matrix)
        clusters = clust.predict(pixel_matrix)

        cluster_centers = clust.cluster_centers_

        cluster_centers_rgb = cluster_centers * 255

        #cluster_centers_hsv = matplotlib.colors.rgb_to_hsv(cluster_centers)
        cluster_centers_hsv = skimage.color.rgb2hsv(cluster_centers)

        clusters_centers_df = pd.DataFrame()

        #   Save HSV
        clusters_centers_df['H'] = cluster_centers_hsv[:, 0]
        clusters_centers_df['S'] = cluster_centers_hsv[:, 1]
        clusters_centers_df['V'] = cluster_centers_hsv[:, 2]

        #   Save RGB
        clusters_centers_df['R'] = cluster_centers_rgb[:, 0]
        clusters_centers_df['G'] = cluster_centers_rgb[:, 1]
        clusters_centers_df['B'] = cluster_centers_rgb[:, 2]

        clusters_centers_df['R'] = clusters_centers_df['R'].apply(np.int64)
        clusters_centers_df['G'] = clusters_centers_df['G'].apply(np.int64)
        clusters_centers_df['B'] = clusters_centers_df['B'].apply(np.int64)

        #   Define buckets for sorting
        clusters_centers_df['bucket'] = 1
        clusters_centers_df.loc[clusters_centers_df['V'] < 0.2, 'bucket'] = 2
        clusters_centers_df.loc[clusters_centers_df['V'] > 0.8, 'bucket'] = 0

        df = pd.DataFrame()
        df['points'] = clusters
        df = df.groupby(by=['points'])['points'].count().rename().reset_index()
        df.columns = ['points', 'count']
        visualization.draw_colors(df, clusters_centers_df, '{}.jpg'.format(filename))
        df.to_csv('{}_data.csv'.format(filename), index=False)
        print('Done')


def generate_clustered_images(image, pixel_matrix, clust, filename):
    #   Scale RGB colors
    pixel_matrix = preprocessing.preprocess_pixel_matrix(pixel_matrix)
    clusters = clust.predict(pixel_matrix)

    cluster_centers = clust.cluster_centers_
    cluster_centers_rgb = cluster_centers * 255
    #cluster_centers_hsv = matplotlib.colors.rgb_to_hsv(cluster_centers)
    cluster_centers_hsv = skimage.color.rgb2hsv(cluster_centers)

    clusters_centers_df = pd.DataFrame()

    #   Save HSV
    clusters_centers_df['H'] = cluster_centers_hsv[:, 0]
    clusters_centers_df['S'] = cluster_centers_hsv[:, 1]
    clusters_centers_df['V'] = cluster_centers_hsv[:, 2]

    #   Save RGB
    clusters_centers_df['R'] = cluster_centers_rgb[:, 0]
    clusters_centers_df['G'] = cluster_centers_rgb[:, 1]
    clusters_centers_df['B'] = cluster_centers_rgb[:, 2]

    clusters_centers_df['R'] = clusters_centers_df['R'].apply(np.int64)
    clusters_centers_df['G'] = clusters_centers_df['G'].apply(np.int64)
    clusters_centers_df['B'] = clusters_centers_df['B'].apply(np.int64)

    visualization.image_to_clustered_colors(image, clusters, clusters_centers_df, filename)
    print('Done')

