from operator import index
import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy.core.fromnumeric import repeat
import pandas as pd
import matplotlib
import imageio
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000
from skimage import io, color
import skimage
from tqdm import tqdm
from scipy.sparse import coo_matrix, vstack

def draw_colors(df_points, clusters_centers_df, filename):
    #   Scale color size extracting the percentage
    total_pixels = np.sum(df_points['count'].values)
    colors_proportions = df_points['count'].values / total_pixels * 100 * 100
    colors_proportions = colors_proportions.astype(np.int64)
    print(colors_proportions)
    df_points['proportion'] = colors_proportions
    image_df = pd.DataFrame(columns=['H', 'S', 'V', 'bucket'])
    for ix, row in df_points.iterrows():
        #tmp_color = clusters_centers_df.loc[row['points'], ['H', 'S', 'V', 'bucket']].values
        tmp_color = clusters_centers_df.loc[row['points'], ['H', 'S', 'V']].reindex(['H','S','V','bucket']).values
        tmp_arr = np.repeat([tmp_color], row['proportion'], axis=0)
        tmp_image_df = pd.DataFrame(tmp_arr, columns=['H', 'S', 'V', 'bucket'])
        image_df = pd.concat([image_df, tmp_image_df])

    sorted_hsv = image_df.sort_values(['bucket', 'H', 'S', 'V'], ascending=[True, False, True, True])[['H', 'S', 'V']].values

    #rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)
    rgb = skimage.color.hsv2rgb(sorted_hsv)

    #scipy.misc.imsave(filename, np.repeat([rgb * 255], 500, axis=0))
    io.imsave(filename,np.repeat([rgb * 255], 500, axis=0).astype(np.uint8))


def image_to_clustered_colors(image, clustering_results, clusters_centers, filename):
    shape = image.shape

    lab = pd.DataFrame(clustering_results, columns=['label'])
    new_image = \
    pd.merge(lab.reset_index(), clusters_centers.reset_index(), left_on='label', right_on='index').sort_values(
        by=['index_x'])[['R', 'G', 'B']].values

    new_image = np.reshape(new_image.astype(np.uint8), shape)
    #scipy.misc.imsave(filename, new_image)
    io.imsave(filename, new_image)

    print('Done')

'''
image_df = pd.DataFrame((pixel_matrix * 255).astype(np.int))
image_df['label'] = clust.labels_
image_df = image_df.loc[image_df['label'] == 3, [0, 1, 2]]

side = np.int(np.ceil(np.sqrt(image_df.shape[0])))

c = 0
matrix = None
for i in tqdm(range(0, side)):
    columns = np.zeros(side)
    for j in range(0, side):
        tmp_color = [255, 255, 255]
        if c < image_df.shape[0]:
            tmp_color = image_df.values[c]

        if matrix is None:
            matrix = [tmp_color]
        else:
            matrix = np.append(matrix, [tmp_color], axis=0)

        c += 1
matrix = matrix.reshape((side, side, 3))
matrix = scipy.misc.imresize(matrix,(500, 500), interp='nearest')
'''