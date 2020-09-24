# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:47:40 2020

@author: kolbi
"""

# test hallucinator
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

highlights = np.array([1473, 1513, 1561])
highlights_txt = np.array(['7s', '7c', '7b'])
df = pd.read_pickle(r'D:\pythonTesting\ML-validation-data\data_df.pkl')

x_to_plot = "dF/F0 3FP"
y_to_plot = "Decay 3FP"

p1 = sns.regplot(data=df, x=x_to_plot, y=y_to_plot, fit_reg=False, marker="o", color="skyblue")

for label, row in df.iterrows():
    if row['variant'] in highlights:    
        variant_label = highlights_txt[np.where(highlights == row['variant'])][0]
        p1.text(row[x_to_plot]+0.2, row[y_to_plot], variant_label, horizontalalignment='left', size='medium', color='black', weight='semibold')
        plt.scatter(row[x_to_plot], row[y_to_plot], marker='+', color='black')
