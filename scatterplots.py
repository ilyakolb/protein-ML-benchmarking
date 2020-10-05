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

plt.close('all')
plt.figure(figsize = [5,5])
highlights = np.array([1473, 1513, 1561])
highlights_txt = np.array(['7s', '7c', '7b'])
df = pd.read_pickle(r'D:\pythonTesting\ML-validation-data\data_df.pkl')

x_to_plot_gt = "dF/F0 160FP"
y_to_plot_gt = "Decay 160FP"
x_to_plot_ML = 'predict_' + x_to_plot_gt
y_to_plot_ML = 'predict_' + y_to_plot_gt

x_to_plot_naive = 'naive_' + x_to_plot_gt
y_to_plot_naive = 'naive_' + y_to_plot_gt

# if plotting decay, small = better
sort_ascending_x = True # "Decay" not in x_to_plot_gt
sort_ascending_y = True # "Decay" not in y_to_plot_gt

# rank by x and y
df_ranked = df.copy()
df_ranked[x_to_plot_gt] = df_ranked[x_to_plot_gt].rank(ascending = sort_ascending_x)
df_ranked[y_to_plot_gt] = df_ranked[y_to_plot_gt].rank(ascending = sort_ascending_y)
df_ranked[x_to_plot_ML] = df_ranked[x_to_plot_ML].rank(ascending = sort_ascending_x)
df_ranked[y_to_plot_ML] = df_ranked[y_to_plot_ML].rank(ascending = sort_ascending_y)
df_ranked[x_to_plot_naive] = df_ranked[x_to_plot_naive].rank(ascending = sort_ascending_x)
df_ranked[y_to_plot_naive] = df_ranked[y_to_plot_naive].rank(ascending = sort_ascending_y)


p1 = sns.scatterplot(data=df_ranked, x=x_to_plot_gt, y=y_to_plot_gt, marker="o", color=[.7, .7, .7], alpha=0.5)
sns.scatterplot(data=df_ranked, x=x_to_plot_ML, y=y_to_plot_ML, marker="o", color=[0, 0, 0.5], alpha=0.5)
sns.scatterplot(data=df_ranked, x=x_to_plot_naive, y=y_to_plot_naive, marker="o", color=[0.5, 0, 0], alpha=0.5)

for label, row in df_ranked.iterrows():
    if row['variant'] in highlights:    
        variant_label = highlights_txt[np.where(highlights == row['variant'])][0]
        
        # ground truth labels
        p1.text(row[x_to_plot_gt]+.2, row[y_to_plot_gt], variant_label, horizontalalignment='left', size='medium', color='black')
        plt.scatter(row[x_to_plot_gt], row[y_to_plot_gt], marker='+', color='black') 
        
        # ML labels
        p1.text(row[x_to_plot_ML]+.2, row[y_to_plot_ML], variant_label, horizontalalignment='left', size='medium', color='blue')
        plt.scatter(row[x_to_plot_ML], row[y_to_plot_ML], marker='+', color='blue') # ML labels
        
        # niave labels
        p1.text(row[x_to_plot_naive]+.2, row[y_to_plot_naive], variant_label, horizontalalignment='left', size='medium', color='red')
        plt.scatter(row[x_to_plot_naive], row[y_to_plot_naive], marker='+', color='red') # ML labels
        
plt.xlabel(x_to_plot_gt)
plt.ylabel(y_to_plot_gt)

plt.tight_layout()