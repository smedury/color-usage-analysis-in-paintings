from PIL import Image
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from constants import *
from utils import *
import matplotlib
from sklearn.decomposition import PCA
import math
import colorsys

#   Cluster colors

def cluster_colors_rgb(pixel_matrix, n_clusters=10, filename='tmp.jpg'):

    #   Scale RGB colors
    scale_colors = pixel_matrix / 255

    #   Create kmeans, fit and rescale colors
    kmeans = KMeans(n_clusters=n_clusters, verbose=3)
    kmeans.fit(scale_colors)
    cluster_colors = kmeans.cluster_centers_ * 255
    cluster_colors = cluster_colors.astype(np.int)

    df = pd.DataFrame()
    df['points'] = kmeans.labels_

    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']

    #   Scale color size extracting the percentage
    total_pixels = np.sum(df['count'].values)
    colors_proportions = df['count'].values / total_pixels * 100 * 100
    colors_proportions = colors_proportions.astype(np.int)
    print(colors_proportions)

    # Generate colors image
    image_row = None
    for idx, c in enumerate(cluster_colors):
        tmp_arr = np.repeat([c], colors_proportions[idx], axis=0)
        if image_row is None:
            image_row = tmp_arr
        else:
            image_row = np.concatenate((image_row, tmp_arr), axis=0)

    #   Sort colors by HSV
    hsv = matplotlib.colors.rgb_to_hsv(image_row)
    hsv_df = pd.DataFrame(hsv, columns=['h', 's', 'v'])
    hsv_rounded_df = hsv_df.round(2)
    hsv_rounded_df.columns = ['h_rounded', 's_rounded', 'v_rounded']

    hsv_df = pd.concat([hsv_df, hsv_rounded_df], axis=1)
    sorted_hsv = hsv_df.sort_values(['h_rounded', 'v_rounded', 's_rounded'], ascending=[True, True, True])[
        ['h', 's', 'v']].values

    rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

    cv2.imwrite(filename, np.repeat([rgb], 500, axis=0))
    print('Done')
    return kmeans

def predict_clusters(pixel_matrix, kmeans, filename):
    #   Scale RGB colors
    scale_colors = pixel_matrix / 255
    clusters = kmeans.predict(scale_colors)
    df = pd.DataFrame()
    df['points'] = clusters

    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']

    #   Scale color size extracting the percentage
    total_pixels = np.sum(df['count'].values)
    colors_proportions = df['count'].values / total_pixels * 100 * 100
    colors_proportions = colors_proportions.astype(np.int)
    print(colors_proportions)

    colors = (kmeans.cluster_centers_*255).astype(np.int)

    # Generate colors image
    image_row = None
    for idx, c in df.iterrows():
        tmp_arr = np.repeat([colors[df.iloc[idx].points]], colors_proportions[idx], axis=0)
        if image_row is None:
            image_row = tmp_arr
        else:
            image_row = np.concatenate((image_row, tmp_arr), axis=0)

    #   Sort colors by HSV
    hsv = matplotlib.colors.rgb_to_hsv(image_row)
    hsv_df = pd.DataFrame(hsv, columns=['h', 's', 'v'])
    hsv_rounded_df = hsv_df.round(2)
    hsv_rounded_df.columns = ['h_rounded', 's_rounded', 'v_rounded']

    hsv_df = pd.concat([hsv_df, hsv_rounded_df], axis=1)
    sorted_hsv = hsv_df.sort_values(['h_rounded', 'v_rounded', 's_rounded'], ascending=[True, True, True])[
        ['h', 's', 'v']].values

    rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

    cv2.imwrite(filename, np.repeat([rgb], 500, axis=0))
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
fitted_model = cluster_colors_rgb(stacked_images, 100, filename='{}/colors_{}.jpg'.format(OUTPUT_FOLDER, 'total'))


for year in data['year'].unique():
    stacked_images = None
    for idx, row in data[data['year'] == year].iterrows():
        img = load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
        reshaped = img.reshape((-1, 3))

        if stacked_images is None:
            stacked_images = reshaped
        else:
            stacked_images = np.concatenate((stacked_images, reshaped))
    predict_clusters(stacked_images, fitted_model, filename='{}/colors_{}.jpg'.format(OUTPUT_FOLDER, year))
