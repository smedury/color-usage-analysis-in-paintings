from sklearn.cluster import KMeans, AgglomerativeClustering
from constants import *
import pandas as pd
import numpy as np
from src.visualization import visualization
import matplotlib
from src.preprocessing import preprocessing
from src.commons import utils
from spherecluster import SphericalKMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm

def cluster_colors_rgb(pixel_matrix, n_clusters=10, filename='tmp.jpg'):
    #   Scale RGB colors
    pixel_matrix = preprocessing.preprocess_pixel_matrix(pixel_matrix)
    #   Create clust, fit and rescale colors
    clust = KMeans(n_clusters=n_clusters, verbose=3)
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis

    '''
    # k means determine k
    distortions = []
    K = range(2, 60)
    for k in tqdm(K):
        kmeanModel = KMeans(n_clusters=k).fit(pixel_matrix)
        kmeanModel.fit(pixel_matrix)
        distortions.append(sum(np.min(cdist(pixel_matrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / pixel_matrix.shape[0])

    # Plot the elbow
    ax.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    fig.savefig('{}/elbow.jpg'.format(OUTPUT_FOLDER))  # save the figure to file
    plt.close(fig)  # close the figure
    plt.show()
    # clust = DBSCAN(metric=cie00_distance, n_jobs=4)
    #clust = SphericalKMeans(n_clusters=n_clusters, verbose=3)
    '''
    clust.fit(pixel_matrix)

    cluster_centers = clust.cluster_centers_
    cluster_centers_rgb = cluster_centers*255
    cluster_centers_hsv = matplotlib.colors.rgb_to_hsv(cluster_centers)

    clusters_centers_df = pd.DataFrame()

    #   Save HSV
    clusters_centers_df['H'] = cluster_centers_hsv[:, 0]
    clusters_centers_df['S'] = cluster_centers_hsv[:, 1]
    clusters_centers_df['V'] = cluster_centers_hsv[:, 2]

    #   Save RGB
    clusters_centers_df['R'] = cluster_centers_rgb[:, 0]
    clusters_centers_df['G'] = cluster_centers_rgb[:, 1]
    clusters_centers_df['B'] = cluster_centers_rgb[:, 2]

    clusters_centers_df['R'] = clusters_centers_df['R'].apply(np.int)
    clusters_centers_df['G'] = clusters_centers_df['G'].apply(np.int)
    clusters_centers_df['B'] = clusters_centers_df['B'].apply(np.int)

    df = pd.DataFrame()
    df['points'] = clust.labels_
    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']
    visualization.draw_colors(df, clusters_centers_df, filename)

    color_scheme = pd.DataFrame(cluster_centers_rgb, columns=['R', 'G', 'B'])
    color_scheme.to_csv('{}/color_scheme.csv'.format(OUTPUT_FOLDER))

    return clust


def generate_color_scheme_images(pixel_matrix, clust, filename):
    #   Scale RGB colors
    pixel_matrix = preprocessing.preprocess_pixel_matrix(pixel_matrix)
    clusters = clust.predict(pixel_matrix)

    cluster_centers = clust.cluster_centers_

    cluster_centers_rgb = cluster_centers*255
    cluster_centers_hsv = matplotlib.colors.rgb_to_hsv(cluster_centers)

    clusters_centers_df = pd.DataFrame()

    #   Save HSV
    clusters_centers_df['H'] = cluster_centers_hsv[:, 0]
    clusters_centers_df['S'] = cluster_centers_hsv[:, 1]
    clusters_centers_df['V'] = cluster_centers_hsv[:, 2]

    #   Save RGB
    clusters_centers_df['R'] = cluster_centers_rgb[:, 0]
    clusters_centers_df['G'] = cluster_centers_rgb[:, 1]
    clusters_centers_df['B'] = cluster_centers_rgb[:, 2]

    clusters_centers_df['R'] = clusters_centers_df['R'].apply(np.int)
    clusters_centers_df['G'] = clusters_centers_df['G'].apply(np.int)
    clusters_centers_df['B'] = clusters_centers_df['B'].apply(np.int)

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
    cluster_centers_rgb = cluster_centers*255
    cluster_centers_hsv = matplotlib.colors.rgb_to_hsv(cluster_centers)

    clusters_centers_df = pd.DataFrame()

    #   Save HSV
    clusters_centers_df['H'] = cluster_centers_hsv[:, 0]
    clusters_centers_df['S'] = cluster_centers_hsv[:, 1]
    clusters_centers_df['V'] = cluster_centers_hsv[:, 2]

    #   Save RGB
    clusters_centers_df['R'] = cluster_centers_rgb[:, 0]
    clusters_centers_df['G'] = cluster_centers_rgb[:, 1]
    clusters_centers_df['B'] = cluster_centers_rgb[:, 2]

    clusters_centers_df['R'] = clusters_centers_df['R'].apply(np.int)
    clusters_centers_df['G'] = clusters_centers_df['G'].apply(np.int)
    clusters_centers_df['B'] = clusters_centers_df['B'].apply(np.int)

    visualization.image_to_clustered_colors(image, clusters, clusters_centers_df, filename)
    print('Done')

preprocessing.resize_images()
data = pd.read_csv('{}/data.csv'.format(DATA_FOLDER))
N_CLUSTERS = 20

#   Stack images
stacked_images = None
fitted_model = None
for idx, row in data.iterrows():
    img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
    reshaped = img.reshape((-1, 3))

    if stacked_images is None:
        stacked_images = reshaped
    else:
        stacked_images = np.concatenate((stacked_images, reshaped))

fitted_model = cluster_colors_rgb(stacked_images, N_CLUSTERS,
                                  filename='{}/colors_{}.jpg'.format(OUTPUT_FOLDER, 'total'))

#   Predict clusters
for year in data['year'].unique():
    stacked_images = None
    for idx, row in data[data['year'] == year].iterrows():
        img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
        reshaped = img.reshape((-1, 3))

        if stacked_images is None:
            stacked_images = reshaped
        else:
            stacked_images = np.concatenate((stacked_images, reshaped))
    generate_color_scheme_images(stacked_images, fitted_model, filename='{}/colors_{}'.format(OUTPUT_FOLDER, year))

for idx, row in data.iterrows():
    img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
    reshaped = img.reshape((-1, 3))
    generate_clustered_images(img, reshaped, fitted_model,
                              filename='{}/clustered_images/colors_{}.jpg'.format(OUTPUT_FOLDER, row.title))
