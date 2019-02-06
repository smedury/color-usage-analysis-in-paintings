from constants import *
from src.visualization.visualization import *


def preprocess_pixel_matrix(pixel_matrix):
    new_matrix = matplotlib.colors.rgb_to_hsv(pixel_matrix / 255)
    return new_matrix

def resize_images(data):
    #   Data with sizes etc
    #   data = pd.read_csv('./data/data.csv', quotechar='"')

    for ix, row in tqdm(data.iterrows()):
        image = load_image('{}/{}.jpg'.format(IMAGE_FOLDER, row.title))
        #   Reshape image according to what's in the data
        image_width = row.width_cm * SCALE_FACTOR
        image_height = row.height_cm * SCALE_FACTOR
        res = cv2.resize(image, dsize=(int(image_width), int(image_height)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title), res)
