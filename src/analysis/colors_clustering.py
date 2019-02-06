from sklearn.cluster import KMeans
from constants import *
from src.visualization.visualization import *
import matplotlib
from src.preprocessing import preprocessing
from src.commons import utils


def cluster_colors_rgb(pixel_matrix, n_clusters=10, filename='tmp.jpg'):
    #   Scale RGB colors
    pixel_matrix = preprocessing.preprocess_pixel_matrix(pixel_matrix)
    #   Create clust, fit and rescale colors
    clust = KMeans(n_clusters=n_clusters, verbose=3)
    # clust = DBSCAN(metric=cie00_distance, n_jobs=4)
    clust.fit(pixel_matrix)

    cluster_centers = clust.cluster_centers_
    cluster_centers_rgb = matplotlib.colors.hsv_to_rgb(cluster_centers)

    cluster_colors_df = pd.DataFrame()
    cluster_colors_df['R'] = cluster_centers_rgb[:, 0] * 255
    cluster_colors_df['G'] = cluster_centers_rgb[:, 1] * 255
    cluster_colors_df['B'] = cluster_centers_rgb[:, 2] * 255

    cluster_colors_df['R'] = cluster_colors_df['R'].apply(np.int)
    cluster_colors_df['G'] = cluster_colors_df['G'].apply(np.int)
    cluster_colors_df['B'] = cluster_colors_df['B'].apply(np.int)

    df = pd.DataFrame()
    df['points'] = clust.labels_
    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']
    draw_colors(df, cluster_centers_rgb, filename)

    color_scheme = pd.DataFrame(cluster_centers_rgb, columns=['R', 'G', 'B'])
    color_scheme.to_csv('{}/color_scheme.csv'.format(OUTPUT_FOLDER))

    return clust


def predict_clusters(pixel_matrix, clust, filename):
    #   Scale RGB colors
    pixel_matrix = preprocessing.preprocess_pixel_matrix(pixel_matrix)
    clusters = clust.predict(pixel_matrix)

    cluster_centers = clust.cluster_centers_
    cluster_centers_rgb = matplotlib.colors.hsv_to_rgb(cluster_centers)

    clusters_centers_df = pd.DataFrame()

    #   Save HSV
    clusters_centers_df['H'] = cluster_centers[:, 0]
    clusters_centers_df['S'] = cluster_centers[:, 1]
    clusters_centers_df['V'] = cluster_centers[:, 2]

    #   Save RGB
    clusters_centers_df['R'] = cluster_centers_rgb[:, 0]
    clusters_centers_df['G'] = cluster_centers_rgb[:, 1]
    clusters_centers_df['B'] = cluster_centers_rgb[:, 2]

    #   Define buckets for sorting
    clusters_centers_df['bucket'] = 1
    clusters_centers_df.loc[clusters_centers_df['v'] < 0.2, 'bucket'] = 2
    clusters_centers_df.loc[clusters_centers_df['v'] > 0.8, 'bucket'] = 0

    df = pd.DataFrame()
    df['points'] = clusters
    df = df.groupby(by=['points'])['points'].count().rename().reset_index()
    df.columns = ['points', 'count']
    draw_colors(df, clusters_centers_df, '{}.jpg'.format(filename))
    df.to_csv('{}_data.csv'.format(filename), index=False)
    print('Done')


data = pd.read_csv('{}/data.csv'.format(DATA_FOLDER))
N_CLUSTERS = 50

stacked_images = None
fitted_model = None
for idx, row in data.iterrows():
    img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
    reshaped = img.reshape((-1, 3))

    if stacked_images is None:
        stacked_images = reshaped
    else:
        stacked_images = np.concatenate((stacked_images, reshaped))

fitted_model = cluster_colors_rgb(stacked_images, N_CLUSTERS, filename='{}/colors_{}.jpg'.format(OUTPUT_FOLDER, 'total'))

for year in data['year'].unique():
    stacked_images = None
    for idx, row in data[data['year'] == year].iterrows():
        img = utils.load_image('{}/{}.jpg'.format(IMAGE_RESHAPED, row.title))
        reshaped = img.reshape((-1, 3))

        if stacked_images is None:
            stacked_images = reshaped
        else:
            stacked_images = np.concatenate((stacked_images, reshaped))
    predict_clusters(stacked_images, fitted_model, filename='{}/colors_{}'.format(OUTPUT_FOLDER, year))
