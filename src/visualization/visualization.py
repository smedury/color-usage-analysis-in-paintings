import cv2
import numpy as np
import pandas as pd
import matplotlib
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000
from skimage import io, color
from tqdm import tqdm
from scipy.sparse import coo_matrix, vstack


def draw_colors(df_points, clusters_centers_df, filename):
    #   Scale color size extracting the percentage
    total_pixels = np.sum(df_points['count'].values)
    colors_proportions = df_points['count'].values / total_pixels * 100 * 100
    colors_proportions = colors_proportions.astype(np.int)
    print(colors_proportions)
    df_points['proportion'] = colors_proportions
    image_df = pd.DataFrame(columns=['H', 'S', 'V', 'bucket'])
    for ix, row in df_points.iterrows():
        tmp_color = clusters_centers_df.loc[row['points'], ['H', 'S', 'V', 'bucket']].values
        tmp_arr = np.repeat([tmp_color], row['proportion'], axis=0)
        tmp_image_df = pd.DataFrame(tmp_arr, columns=['H', 'S', 'V', 'bucket'])
        image_df = pd.concat([image_df, tmp_image_df])

    sorted_hsv = image_df.sort_values(['bucket', 'H', 'S', 'V'], ascending=[True, False, True, True])[
        ['H', 'S', 'V']].values

    rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

    cv2.imwrite(filename, np.repeat([rgb * 255], 500, axis=0))


def image_to_clustered_colors(image, clustering_results, clusters_centers, filename):
    shape = image.shape

    lab = pd.DataFrame(clustering_results, columns=['label'])
    new_image = \
    pd.merge(lab.reset_index(), clusters_centers.reset_index(), left_on='label', right_on='index').sort_values(
        by=['index_x'])[['R', 'G', 'B']].values

    new_image = np.reshape(new_image, shape)
    cv2.imwrite(filename, new_image)

    print('Done')
