from constants import *
from src.visualization.visualization import *
from src.commons import utils


def resize_images(data, size=None):
    '''
    Resize the images
    :param data: The dataframe with all the information
    :param size: None or a tuple with width and height of the output images
    :return:
    '''
    if size is not None:
        data['width_cm'] = size[0]
        data['height_cm'] = size[0]

    for ix, row in tqdm(data.iterrows()):
        image = utils.load_image('{}/{}.jpg'.format(IMAGE_FOLDER, row.title))
        #   Reshape image according to what's in the data
        image_width = row['width_cm'] * SCALE_FACTOR
        image_height = row['height_cm'] * SCALE_FACTOR
        res = cv2.resize(image, dsize=(int(image_width), int(image_height)), interpolation=cv2.INTER_CUBIC)
        scipy.misc.imsave('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title), res)



def preprocess_pixel_matrix(pixel_matrix):
    new_matrix = pixel_matrix / 255
    return new_matrix
