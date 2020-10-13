# -*- coding: utf-8 -*-
"""
load model df and annotated df with mutated positions
use to find out how many novel positions are in GC6 dataset

Created on Tue May 12 17:47:40 2020

@author: kolbi
"""

import numpy as np
import pandas as pd


model = 'convunirep' # convunirep or convprobaaseq
predict_type = 'mean+std'
df_save_path = r'D:\pythonTesting\ML-validation-data\data_df_predict=' +predict_type+ '_model=' + model + '.pkl'

data_df = pd.read_pickle(df_save_path)
data_df.set_index('variant')

# load annotated data (used to get position list)
annotated_data_GC3 = pd.read_pickle('data_belongs_shuf_train_GC3.pkl')
annotated_data_GC6 = pd.read_pickle('data_belongs_shuf_train_GC6.pkl')

annotated_data_GC3 = annotated_data_GC3.set_index('variant')
annotated_data_GC6 = annotated_data_GC6.set_index('variant')

all_positions_in_GC3 = []

for p in annotated_data_GC3['position']:
    all_positions_in_GC3.extend(p)
    
all_positions_in_GC3 = np.unique(all_positions_in_GC3)

all_positions_in_GC6 = []

for p in annotated_data_GC6['position']:
    all_positions_in_GC6.extend(p)
all_positions_in_GC6 = np.unique(all_positions_in_GC6)

unique_pos = np.setdiff1d(all_positions_in_GC6, all_positions_in_GC3)
print('Positions in GC6 not in GC3:')
print(unique_pos)

r = pd.merge(data_df, annotated_data_GC6['position'], on='variant')
include_variant = [np.intersect1d(p, all_positions_in_GC3).size == 0 for p in r['position']]

print('Num variants with no positions common to GC3 dataset: {}'.format(sum(include_variant)))

