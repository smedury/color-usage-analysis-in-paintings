from constants import *
from src.visualization.visualization import *
from src.commons import utils

def preprocess_pixel_matrix(pixel_matrix):
    new_matrix = pixel_matrix/255
    return new_matrix

def resize_images():
    #   Data with sizes etc
    data = pd.read_csv('{}/data.csv'.format(DATA_FOLDER), quotechar='"')

    for ix, row in tqdm(data.iterrows()):
        image = utils.load_image('{}/{}.jpg'.format(IMAGE_FOLDER, row.title))
        #   Reshape image according to what's in the data
        image_width = row.width_cm * SCALE_FACTOR
        image_height = row.height_cm * SCALE_FACTOR
        res = cv2.resize(image, dsize=(int(image_width), int(image_height)), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title), res)
