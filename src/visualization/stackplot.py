import pandas as pd
import sys

from pandas.core.arrays import integer
sys.path.append(".")
from src.constants import *
from src.commons import utils
import matplotlib.pyplot as plt
import numpy as np
from webcolors import rgb_to_name

from scipy.spatial import KDTree

def riverdiagram(num_colors):
    images = pd.read_csv('{}/data.csv'.format(DATA_FOLDER))
    x=range(0,19)
    years_temp = images['year'].unique()
    years=[]
    for y in years_temp:
        if(y is not np.NaN and str(y).isnumeric() and int(y)>1900 and int(y)<2021):
            years.append(y)
    years = sorted(years)
    colors=[]
    all_data = pd.DataFrame(columns=years)
    for year in years:
        new_data = pd.read_csv('{}/years_proportions/{}.csv'.format(OUTPUT_FOLDER, year), names=['perc','R','G','B','H','S','V'], skiprows=[0])
        new_data_improved = new_data.apply(pd.to_numeric, args=('coerce',)).dropna()
        percentages = new_data_improved['perc'].astype('float')
        all_data[year]=np.absolute(percentages.values[0:num_colors])
        if(len(colors)==0):
            colors_rgb = pd.DataFrame()
            colors_rgb['R'] = new_data['R'].astype(np.int16)
            colors_rgb['G'] = new_data['G'].astype(np.int16)
            colors_rgb['B'] = new_data['B'].astype(np.int16)

            for (r,g,b) in zip(colors_rgb['R'],colors_rgb['G'],colors_rgb['B']):
                colors.append('#%02x%02x%02x' % (r, g, b))

    plt.figure(figsize=(8, 6))
    plt.stackplot(years,all_data.values,baseline='zero',colors=colors)
    #plt.show()
    plt.savefig('{}/riverdiagram-totalcolors-usage.png'.format(OUTPUT_FOLDER))