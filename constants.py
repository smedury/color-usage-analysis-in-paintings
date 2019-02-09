import os

SCALE_FACTOR = 1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

DATA_FOLDER = '{}/data'.format(ROOT_DIR)
IMAGE_FOLDER = '{}/images'.format(DATA_FOLDER)
IMAGE_RESHAPED = '{}/images_reshaped'.format(DATA_FOLDER)
OUTPUT_FOLDER = '{}/output/kmeans'.format(DATA_FOLDER)
CLUSTER_CHART_WIDTH = 100