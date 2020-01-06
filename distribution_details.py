# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 22:46:07 2019

@author: Haribaskar.d
"""

from sklearn.model_selection import KFold
import numpy as np


def distribution_check_detail(data=None, features= None, target = None, fold_num = None):
    '''
    It calculates the mean and std for each feature columns.
    if fold_num = None : It considers the whole data as a single fold
    else : It splits the data to corresponding folds and gets the average mean & std for each feature columns.
    '''
    if fold_num:
        # Kfold the data:
        K_data = list(KFold(n_splits = fold_num, shuffle=True, random_state=123).split(data[features]))
        # creating an empty dictionary:
        dist = {i:{'mean':[],'std':[]} for i in features}
        for j in range(len(K_data)):
            # subsetting each fold data:
            data_subset = data.iloc[list(K_data[j][0])][features]
            # getting the distribution of all the features - for each fold:
            for i in features:
                dist[i]['mean'].append(np.mean(data_subset[i]))
                dist[i]['std'].append(np.std(data_subset[i]))
        # aggregating the distribution for all the folds:
        distribution = {i:{'mean':np.mean(dist[i]['mean']),'std':np.mean(dist[i]['std'])} for i in list(data[features].columns)}
        
    else:
        distribution = {i:{'mean':np.mean(data[i]),'std':np.std(data[i]),'missing%':(data[i].isna().sum()/len(data))*100} for i in list(data[features].columns)}
    return(distribution)





    
        
    