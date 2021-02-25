import pandas as pd
import sys
sys.path.append(".")
from src.constants import *
from src.commons import utils
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('{}/data.csv'.format(DATA_FOLDER))

Path('{}/years_proportions'.format(OUTPUT_FOLDER)).mkdir(parents=True, exist_ok=True)

for year in data['year'].unique():
    full_data = None
    try:
        for idx, row in data[data['year'] == year].iterrows():
            temp_data = pd.read_csv('{}/images_schemes/{}_data.csv'.format(OUTPUT_FOLDER, row.title))
            if full_data is None:
                full_data = temp_data
            else:
                full_data = pd.concat([full_data, temp_data], axis=0)
        if full_data is not None:
            full_data = full_data.groupby(['points']).sum().reset_index()
            full_data['perc'] = full_data['count'] / full_data['count'].sum()

            #   Merge colors
            color_scheme = pd.read_csv('{}/color_scheme.csv'.format(OUTPUT_FOLDER))
            full_data = pd.merge(full_data, color_scheme, how='inner', left_on='points', right_on=color_scheme.columns[0])

            full_data['R'] = full_data['R'].apply(np.int64)
            full_data['G'] = full_data['G'].apply(np.int64)
            full_data['B'] = full_data['B'].apply(np.int64)

            full_data.to_csv('{}/years_proportions/{}.csv'.format(OUTPUT_FOLDER, year),
                         columns=['perc', 'R', 'G', 'B', 'H', 'S', 'V'],
                         index=False)
    except {KeyError}:
        print('Key not found')
