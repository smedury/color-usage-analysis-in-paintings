import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from constants import *
from utils import *




#   Data with sizes etc
data = pd.read_csv('./data/data.csv', quotechar='"')

for ix, row in tqdm(data.iterrows()):
    image = load_image('{}/{}.jpg'.format(IMAGE_FOLDER, row.title))
    #   Reshape image according to what's in the data
    image_width = row.width_cm * SCALE_FACTOR
    image_height = row.height_cm * SCALE_FACTOR
    res = cv2.resize(image, dsize=(int(image_width), int(image_height)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title), res)

print('Done')
