import pandas as pd
import sys

from pandas.core.arrays import integer
sys.path.append(".")
from src.constants import *
from src.commons import utils
import matplotlib.pyplot as plt
import numpy as np

images = pd.read_csv('{}/images/data.csv'.format(DATA_FOLDER))
years = images['year'].unique() 
all_data = pd.DataFrame(index=['year'],columns=["perc","color"])
for year in years:
    if (not(year is np.NAN)) :
        new_data = pd.read_csv('{}/years_proportions/{}.csv'.format(OUTPUT_FOLDER, year), names=['perc','R','G','B','H','S','V'], skiprows=[0])
        all_data['year'] = [new_data['perc'],new_data['R','G','B','H','S','V']]
        #if all_data is None:
        
        #else:
        #   all_data = pd.concat([all_data, new_data], axis=0)

fig, ax = plt.subplots()
print(all_data.values)
print(all_data.keys())
ax.stackplot(years, all_data["perc"].values,
             labels=all_data.keys())
ax.legend(loc='upper left')
ax.set_title('Pollock use of color')
ax.set_xlabel('Year')
ax.set_ylabel('Colors')

plt.show()
