from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from skimage import io
from tqdm import tqdm


def load_image(filename):
    #img = skimage.io.imread(filename)
    img = io.imread(filename)
    return img


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
