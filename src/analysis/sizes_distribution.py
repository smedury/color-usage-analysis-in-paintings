import pandas as pd
import sys
sys.path.append(".")
from src.constants import *
import matplotlib.pyplot as plt
data = pd.read_csv('{}/images/data.csv'.format(DATA_FOLDER))
data['Canvas size in square meters'] = data['height_cm']/100*data['width_cm']/100
data.boxplot(['Canvas size in square meters'])
plt.show()
print('OK')