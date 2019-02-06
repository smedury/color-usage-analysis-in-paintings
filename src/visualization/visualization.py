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

    image_df = pd.DataFrame(columns=['H', 'S', 'V', 'bucket'])
    for ix, row in clusters_centers_df.iterrows():
        tmp_color = row[['H', 'S', 'V', 'bucket']].values
        tmp_arr = np.repeat([tmp_color], colors_proportions[ix], axis=0)
        tmp_image_df = pd.DataFrame(tmp_arr, columns=['H', 'S', 'V', 'bucket'])
        image_df = pd.concat([image_df, tmp_image_df])

    sorted_hsv = image_df.sort_values(['bucket', 'H', 'S', 'V'], ascending=[True, False, True, True])[
        ['H', 'S', 'V']].values

    rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

    cv2.imwrite(filename, np.repeat([rgb * 255], 500, axis=0))


