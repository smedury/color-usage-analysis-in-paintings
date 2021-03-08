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

from src.scraper.scraper_rothko import *
import numpy as np
import pandas as pd
from src.constants import *
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from pathlib import Path

if __name__ == "__main__":

    process.crawl(PaintingsSpider)
    process.start()  # the script will block here until the crawling is finished
    print('hello')

    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_RESHAPED).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv('{}/data.csv'.format(DATA_FOLDER))

    preprocessing.resize_images(data, size=(200,200))
    N_CLUSTERS = 20

    #   Stack images
    stacked_images = None
    fitted_model = None
    for idx, row in data.iterrows():
        img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
        reshaped = img.reshape((-1, 3))

        if stacked_images is None:
            stacked_images = reshaped
            #break
        else:
            stacked_images = np.concatenate((stacked_images, reshaped))

    fitted_model = cluster_colors_rgb(stacked_images, N_CLUSTERS,
                                      filename='{}/colors_{}.jpg'.format(OUTPUT_FOLDER, 'total'))

    #   Predict clusters
    for year in data['year'].unique():
        stacked_images = None
        for idx, row in data[data['year'] == year].iterrows():
            img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
            reshaped = img.reshape((-1, 3))

            if stacked_images is None:
                stacked_images = reshaped
            else:
                stacked_images = np.concatenate((stacked_images, reshaped))
        generate_color_scheme_images(stacked_images, fitted_model, filename='{}/colors_{}'.format(OUTPUT_FOLDER, year))

    #   Predict clusters
    for idx, row in data.iterrows():
        stacked_images = None
        img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
        reshaped = img.reshape((-1, 3))

        Path('{}/images_schemes'.format(OUTPUT_FOLDER)).mkdir(parents=True, exist_ok=True)

        generate_color_scheme_images(reshaped, fitted_model,
                                     filename='{}/images_schemes/{}'.format(OUTPUT_FOLDER, row.title))

    for idx, row in data.iterrows():
        img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
        reshaped = img.reshape((-1, 3))
        Path('{}/clustered_images'.format(OUTPUT_FOLDER)).mkdir(parents=True, exist_ok=True)
        generate_clustered_images(img, reshaped, fitted_model,
                                  filename='{}/clustered_images/colors_{}.jpg'.format(OUTPUT_FOLDER, row.title))

    occurences_per_year()