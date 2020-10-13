# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:47:40 2020

@author: kolbi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.close('all')
highlights = np.array([1473, 1513, 1561])
highlights_txt = np.array(['7s', '7c', '7b'])

model = 'convunirep' # convunirep or convprobaaseq
predict_type = 'mean+std'
df_save_path = r'D:\pythonTesting\ML-validation-data\data_df_predict=' +predict_type+ '_model=' + model + '.pkl'

df = pd.read_pickle(df_save_path)

FPnums = [1,3,10,160]

for FPnum in FPnums:
    plt.figure(figsize = [5,5])

    x_to_plot_gt = "dF/F0 {}FP".format(FPnum)
    y_to_plot_gt = "Decay {}FP".format(FPnum)
    x_to_plot_ML = 'predict_' + x_to_plot_gt
    y_to_plot_ML = 'predict_' + y_to_plot_gt
    
    x_to_plot_naive = 'naive_' + x_to_plot_gt
    y_to_plot_naive = 'naive_' + y_to_plot_gt
    
    # for decay, higher rank = slower. for dff, higher rank = more sensitive
    sort_ascending_x = True 
    sort_ascending_y = True 
    
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
    
    # box plots of rank differences (ML vs naive) of top hits
    fig = plt.figure()
    fig.suptitle(x_to_plot_gt + " and " + y_to_plot_gt)
    top_percents = [10, 20, 50, 100] # percent above which to do cutoff
    for i,top_percent in enumerate(top_percents):
        
        ax = fig.add_subplot(2,2,i+1)
        top_ranknum = int(len(df_ranked) * (1 - top_percent/100))
        
        df_ranked_top = df_ranked[df_ranked[x_to_plot_gt] > top_ranknum].copy()
    
        
        df_ranked_top['ML vs GT'] = np.abs(df_ranked_top[x_to_plot_gt] - df_ranked_top[x_to_plot_ML]) + np.abs(df_ranked_top[y_to_plot_gt] - df_ranked_top[y_to_plot_ML])
        df_ranked_top['naive vs GT'] = np.abs(df_ranked_top[x_to_plot_gt] - df_ranked_top[x_to_plot_naive]) + np.abs(df_ranked_top[y_to_plot_gt] - df_ranked_top[y_to_plot_naive])
        df_comp = df_ranked_top.melt(value_vars=['ML vs GT', 'naive vs GT'], var_name = 'model', value_name = 'rank difference' )
        ax = sns.swarmplot(x='model',  y='rank difference', data=df_comp, color=".25", alpha=0.5)
        ax = sns.boxplot(x='model',  y='rank difference', data=df_comp)
        ax.set_title("top {}%".format(top_percent))
        ax.set_xlabel('')
        plt.tight_layout()