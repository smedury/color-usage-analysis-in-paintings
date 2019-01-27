from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from constants import *
from utils import *

#   Cluster colors
img = load_image('{}/alchemy.jpg'.format(IMAGE_RESHAPED))
reshaped = img.reshape((-1, 3))/255

kmeans = KMeans(n_clusters=50, verbose=3)

print('Done')
