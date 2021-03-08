from src.visualization.stackplot import riverdiagram
import skimage
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from src.analysis.occurences_per_year import occurences_per_year
from src.commons.utils import *
import src.scraper.custom_image_pipeline as custom_image_pipeline
from src.preprocessing import preprocessing
from src.visualization import visualization
from src.analysis.colors_clustering_kmeans import *
from src.analysis.occurences_per_year import *

from src.scraper.scraper_kahlo import *
import numpy as np
import pandas as pd
from src.constants import *
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from pathlib import Path

if __name__ == "__main__":
    riverdiagram(16)