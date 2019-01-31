import cv2
import numpy as np
import pandas as pd
import matplotlib
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976, delta_e_cie2000


def load_image(filename):
    img = cv2.imread(filename)
    return img

def draw_colors(df_points, clusters_centers, filename):
    #   Scale color size extracting the percentage
    total_pixels = np.sum(df_points['count'].values)
    colors_proportions = df_points['count'].values / total_pixels * 100 * 100
    colors_proportions = colors_proportions.astype(np.int)
    print(colors_proportions)

    colors = (clusters_centers * 255).astype(np.int)

    # Generate colors image
    image_row = None
    for idx, c in df_points.iterrows():
        tmp_arr = np.repeat([colors[df_points.iloc[idx].points]], colors_proportions[idx], axis=0)
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

    '''
    # Reference color.
    color1 = LabColor(lab_l=0.9, lab_a=16.3, lab_b=-2.22)
    # Color to be compared to the reference.
    color2 = LabColor(lab_l=0.7, lab_a=14.2, lab_b=-1.80)
    # This is your delta E value as a float.
    delta_e = delta_e_cie1976(color1, color2)
    '''
    rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

    cv2.imwrite(filename, np.repeat([rgb], 500, axis=0))
