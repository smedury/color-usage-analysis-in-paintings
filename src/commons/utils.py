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

def load_image(filename):
    img = cv2.imread(filename)
    return img

'''
def draw_colors(df_points, clusters_centers_df, filename):
    #   Scale color size extracting the percentage
    total_pixels = np.sum(df_points['count'].values)
    colors_proportions = df_points['count'].values / total_pixels * 100 * 100
    colors_proportions = colors_proportions.astype(np.int)
    print(colors_proportions)

    colors = (clusters_centers_df.loc[:,['r', 'g', 'b']] * 255).astype(np.int).values

    # Generate colors image
    image_row = None
    for idx, c in df_points.iterrows():
        tmp_arr = np.repeat([colors[df_points.iloc[idx].points]], colors_proportions[idx], axis=0)
        if image_row is None:
            image_row = pd.DataFrame(tmp_arr)
            image_row['bucket'] = clusters_centers_df[clusters_centers_df[]c['points']]
        else:
            image_row = np.concatenate((image_row, tmp_arr), axis=0)

    #   Sort colors by HSV
    hsv = matplotlib.colors.rgb_to_hsv(image_row)
    hsv_df = pd.DataFrame(hsv, columns=['h', 's', 'v'])
    hsv_rounded_df = hsv_df.round(2)
    hsv_rounded_df.columns = ['h_rounded', 's_rounded', 'v_rounded']

    hsv_df = pd.concat([hsv_df, hsv_rounded_df], axis=1)
    for b in clusters_centers_df['bucket'].unique():
        tmp_bucket = clusters_centers_df[clusters_centers_df['bucket']==b]
        sorted_hsv = tmp_bucket.sort_values(['h', 's', 'v'], ascending=[True, True, True])[
            ['h', 's', 'v']].values


    rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

    cv2.imwrite(filename, np.repeat([rgb], 500, axis=0))
'''

def draw_colors(df_points, clusters_centers_df, filename):
    #   Scale color size extracting the percentage
    total_pixels = np.sum(df_points['count'].values)
    colors_proportions = df_points['count'].values / total_pixels * 100 * 100
    colors_proportions = colors_proportions.astype(np.int)
    print(colors_proportions)

    image_df = pd.DataFrame(columns=['h','s','v','bucket'])
    for ix, row in clusters_centers_df.iterrows():
        tmp_color = row[['h', 's', 'v' ,'bucket']].values
        tmp_arr = np.repeat([tmp_color], colors_proportions[ix], axis=0)
        tmp_image_df = pd.DataFrame(tmp_arr, columns=['h', 's', 'v', 'bucket'])
        image_df = pd.concat([image_df,tmp_image_df])

    sorted_hsv = image_df.sort_values(['bucket','h', 'v', 's'], ascending=[True,False,True,True])[
        ['h', 's', 'v']].values

    rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

    cv2.imwrite(filename, np.repeat([rgb*255], 500, axis=0))


def rgb_to_lab(pixel_matrix):
    lab_pixel_matrix = []

    for rgb in tqdm(pixel_matrix):
        rgb = sRGBColor(rgb[0], rgb[1], rgb[2])
        lab = convert_color(rgb, LabColor)

        lab_pixel_matrix.append([lab.lab_l, lab.lab_a, lab.lab_b])

    return lab_pixel_matrix

def cie00_distance(lab1, lab2):
    # This is your delta E value as a float.
    delta_e = delta_e_cie2000(LabColor(lab1[0], lab1[1], lab1[2]), LabColor(lab2[0], lab2[1], lab2[2]))
    return delta_e
