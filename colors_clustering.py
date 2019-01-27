from PIL import Image
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from constants import *
from utils import *
import matplotlib
from sklearn.decomposition import PCA
import math
import colorsys

#   Cluster colors
img = load_image('{}/ocean greyness.jpg'.format(IMAGE_RESHAPED))
reshaped = img.reshape((-1, 3)) / 255

kmeans = KMeans(n_clusters=10, verbose=3)
kmeans.fit(reshaped)
cluster_colors = kmeans.cluster_centers_ * 256
cluster_colors = cluster_colors.astype(np.int)

df = pd.DataFrame()
df['points'] = kmeans.labels_

df = df.groupby(by=['points'])['points'].count().rename().reset_index()
df.columns = ['points', 'count']

#   Scale the numbers
total_pixels = np.sum(df['count'].values)
colors_proportions = df['count'].values / total_pixels * 100 * 100
colors_proportions = colors_proportions.astype(np.int)
print(colors_proportions)

image_row = None
for idx, c in enumerate(cluster_colors):
    tmp_arr = np.repeat([c], colors_proportions[idx], axis=0)
    if image_row is None:
        image_row = tmp_arr
    else:
        image_row = np.concatenate((image_row, tmp_arr), axis=0)
cv2.imwrite('tmp2.jpg', np.repeat([image_row], 500, axis=0))
#   Repeat the row for CLUSTER_CHART_WIDTH times

hsv = matplotlib.colors.rgb_to_hsv(image_row)
hsv_df = pd.DataFrame(hsv, columns=['h', 's', 'v'])
hsv_rounded_df = hsv_df.round(2)
hsv_rounded_df.columns = ['h_rounded', 's_rounded', 'v_rounded']

hsv_df = pd.concat([hsv_df, hsv_rounded_df], axis=1)
sorted_hsv = hsv_df.sort_values(['h_rounded', 'v_rounded', 's_rounded'], ascending=[True, False, True])[
    ['h', 's', 'v']].values

rgb = matplotlib.colors.hsv_to_rgb(sorted_hsv)

cv2.imwrite('tmp2.jpg', np.repeat([rgb], 500, axis=0))
print('Done')
