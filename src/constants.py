from pathlib import Path

SCALE_FACTOR = 1
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
#ROOT_DIR = os.path.dirname(os.path.abspath('/home/smedury/GitRepos/pollock-analysis/src')) # This is your Project Root
#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
path = Path(__file__)
ROOT_DIR = path.parent.parent # This is your Project Root

DATA_FOLDER = '{}/data'.format(ROOT_DIR)
IMAGE_FOLDER = '{}/images'.format(DATA_FOLDER)
IMAGE_RESHAPED = '{}/images_reshaped'.format(DATA_FOLDER)
OUTPUT_FOLDER = '{}/output/kmeans'.format(DATA_FOLDER)
CLUSTER_CHART_WIDTH = 100