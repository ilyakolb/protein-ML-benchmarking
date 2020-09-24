# -*- coding: utf-8 -*-
"""
Created on 9/21/20

@author: kolbi

predict performance of gcamp6 dataset from gcamp3 data with and w/o machine learning prior
"""

# evaluate ML predictions based on testing budget

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fraction_top_score(top_pred, gt_variants, n_to_predict):
	"""
	evaluate what fraction of top true variants are correctly predicted by model
	@inputs:
		pred_variants (int array): ranked (ascending) variant names predicted top variants
		gt_variants (int array): ranked (ascending) variant names ground truth variants
		n_to_predict (int): number of top constructs in ground truth data to predict
	@ output:
		fraction (float): fraction of top constructs in trueVals that are also in the top of predVals
	"""

	top_gt_buffer = gt_variants[:n_to_predict]

	return len(np.intersect1d(top_pred, top_gt_buffer)) / len(top_gt_buffer)

def performance_given_budget(data, var_predict, b, top_N_to_predict, nTimes):
    '''
    performance_given_budget: calculate performance metrics given a screening budget of b variants
    inputs:
        data (pandas array): dataframe with ground truth and predictions
        var_predict (str): e.g. 'dF/F0 3AP'
        b (int): test budget
        top_N_to_predict (int): how many top variants in ground truth data to predict
        nTimes (int): num times to run the random model
    
    outputs:
        mean and std of ML model, naive model, and random sampling
        
    '''
    scores_ML = []
    scores_random = []
    scores_naive = []
    for i in np.arange(nTimes):
        
        reverse_sort = 'Decay' in var_predict # reverse sort (descending) if df/f. otherwise, regular sort
        gt_ranked_variants = data.sort_values(by=var_predict, ascending=reverse_sort)['variant'].values
        predict_ranked_variants= data.sort_values(by='predict_' + var_predict, ascending=reverse_sort)['variant'].values
        
        top_randsample_variants = data.sample(frac=1)['variant'][:b]
        
        # strategy for choosing top hits
        top_predict_ranked_variants = predict_ranked_variants[:b]
        
        # machine learning model
        s = fraction_top_score(top_predict_ranked_variants, gt_ranked_variants, top_N_to_predict)
		
        # naive machine learning model (FOR NOW, = RANDOM MODEL)
        s_naive = fraction_top_score(top_randsample_variants, gt_ranked_variants, top_N_to_predict)
		
        # random model
        s_random = fraction_top_score(top_randsample_variants, gt_ranked_variants, top_N_to_predict)

        scores_ML.append(s)
        scores_naive.append(s_naive)
        scores_random.append(s_random)

	# print(len(avgScores))
    return (np.mean(scores_ML), np.std(scores_ML), np.mean(scores_naive), np.std(scores_naive), np.mean(scores_random), np.std(scores_random))


num_gt_data_to_predict = 20 # predicting this many of the top ground truths
budget = np.arange(20,655,20) # test budget
nTimesToRun = 10

all_vars_to_predict = ['dF/F0 1FP', 'dF/F0 3FP', 'dF/F0 10FP', 'dF/F0 160FP', 'Decay 1FP', 'Decay 3FP', 'Decay 10FP', 'Decay 160FP']

# assume labels in data_df are in the same order as in npz files
data = np.load(r"D:\pythonTesting\screening_spreadsheets\vinay_prediction_data/aaseq-f1-predictions-allfolds.npz")

data_df = pd.read_csv(r"D:\pythonTesting\screening_spreadsheets\vinay_prediction_data/ground-truth-with-variant-names.csv")

f1 = {'data_file': data['data_file'][0], 'b_train': data['b_train'], 'y': data['y'], 'yhat_mean': data['yhat_mean'], 'yhat_stddev': data['yhat_stddev']}

plt.close('all')
plt.figure(figsize=[8,4])


y_mean = np.mean(f1['y'], 2)
yhat_mean_mean = np.mean(f1['yhat_mean'], 2)
yhat_stddev_mean = np.mean(f1['yhat_stddev'], 2)

data_df['predict_dF/F0 1FP'] = f1['yhat_mean'][0,:,0].T
data_df['predict_dF/F0 3FP'] = f1['yhat_mean'][0,:,1].T
data_df['predict_dF/F0 10FP'] = f1['yhat_mean'][0,:,2].T
data_df['predict_dF/F0 160FP'] = f1['yhat_mean'][0,:,3].T
data_df['predict_Decay 1FP'] = f1['yhat_mean'][0,:,4].T
data_df['predict_Decay 3FP'] = f1['yhat_mean'][0,:,5].T
data_df['predict_Decay 10FP'] = f1['yhat_mean'][0,:,6].T
data_df['predict_Decay 160FP'] = f1['yhat_mean'][0,:,7].T

s=1 # subplot index

for var_to_predict in all_vars_to_predict:
    mean_ML = np.zeros(len(budget))
    std_ML = np.zeros(len(budget))
    mean_naive = np.zeros(len(budget))
    std_naive = np.zeros(len(budget))
    mean_random = np.zeros(len(budget))
    std_random = np.zeros(len(budget))
    
    for i,b in enumerate(budget):
    	mean_ML[i], std_ML[i], mean_naive[i], std_naive[i], mean_random[i], std_random[i] = [f for f in performance_given_budget(data_df, var_to_predict, b, num_gt_data_to_predict, nTimesToRun)]
    
    ax = plt.subplot(2,4,s)
    plt.plot(budget, mean_ML, 'b-')
    # plt.fill_between(budget, mean_ML-std_ML, mean_ML+std_ML, color=[.5, 0, 0], alpha=0.6)
    plt.plot(budget, mean_random, color=[.5, 0, 0])
    plt.fill_between(budget, mean_random-std_random, mean_random+std_random, color=[.5, 0, 0], alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.show()
    plt.title(var_to_predict, fontsize=10)
    plt.tight_layout()
    
    if s >= 5 and s <= 8:
        plt.xlabel('test budget', fontsize=8)
    s+=1