from PIL import Image
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from constants import *
from utils import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib
from sklearn.decomposition import PCA
import math
import colorsys

def preprocess_pixel_matrix(pixel_matrix):

    new_matrix = matplotlib.colors.rgb_to_hsv(pixel_matrix/255)
    return new_matrix

def cluster_colors_rgb(pixel_matrix, n_clusters=10, filename='hello'):

    #   Scale RGB colors
    pixel_matrix = preprocess_pixel_matrix(pixel_matrix)

    #   Create kmeans, fit and rescale colors
    kmeans = KMeans(n_clusters=n_clusters, verbose=3)
    kmeans.fit(pixel_matrix)

    cluster_centers = kmeans.cluster_centers_
    cluster_centers_rgb = matplotlib.colors.hsv_to_rgb(cluster_centers)

    cluster_colors_df = pd.DataFrame()
    cluster_colors_df['R'] = cluster_centers_rgb[:, 0]*255
    cluster_colors_df['G'] = cluster_centers_rgb[:, 1]*255
    cluster_colors_df['B'] = cluster_centers_rgb[:, 2]*255

    cluster_colors_df['R'] = cluster_colors_df['R'].apply(np.int)
    cluster_colors_df['G'] = cluster_colors_df['G'].apply(np.int)
    cluster_colors_df['B'] = cluster_colors_df['B'].apply(np.int)

    df = pd.DataFrame()
    df['points'] = kmeans.labels_
    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']
    draw_colors(df, cluster_centers_rgb, filename)

    color_scheme = pd.DataFrame(cluster_centers_rgb, columns=['R', 'G', 'B'])
    color_scheme.to_csv('{}/color_scheme.csv'.format(OUTPUT_FOLDER))

    return kmeans



def predict_clusters(pixel_matrix, kmeans, filename):
    #   Scale RGB colors
    pixel_matrix = preprocess_pixel_matrix(pixel_matrix)
    clusters = kmeans.predict(pixel_matrix)

    print(clusters)

    cluster_centers = kmeans.cluster_centers_
    cluster_centers_rgb = matplotlib.colors.hsv_to_rgb(cluster_centers)

    df = pd.DataFrame()
    df['points'] = clusters
    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']
    draw_colors(df, cluster_centers_rgb, '{}.jpg'.format(filename))
    df.to_csv('{}_data.csv'.format(filename), index=False)
    print('Done')



data = pd.read_csv('{}/data.csv'.format(DATA_FOLDER))


stacked_images = None
fitted_model = None
for idx, row in data.iterrows():
    img = load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
    reshaped = img.reshape((-1, 3))

    if stacked_images is None:
        stacked_images = reshaped
    else:
        stacked_images = np.concatenate((stacked_images, reshaped))
fitted_model = cluster_colors_rgb(stacked_images, 50, filename='{}/colors_{}.jpg'.format(OUTPUT_FOLDER, 'total'))


for year in data['year'].unique():
    stacked_images = None
    for idx, row in data[data['year'] == year].iterrows():
        img = load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
        reshaped = img.reshape((-1, 3))

        if stacked_images is None:
            stacked_images = reshaped
        else:
            stacked_images = np.concatenate((stacked_images, reshaped))
    predict_clusters(stacked_images, fitted_model, filename='{}/colors_{}'.format(OUTPUT_FOLDER, year))
