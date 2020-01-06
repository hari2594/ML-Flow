
##########################################################################################################################
##  Importing the necessary libraries:                                                                                  ##
##########################################################################################################################

import os
import numpy as np
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as misno
import time
import math
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import logging
logger = logging.getLogger('ftpuploader')
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch


###########################################################################################################################
## This Class is used to build a GBM | Randomforest h2o model and save the model in a POJO or a JAR file.                ##
## The model can be build using the default parameters or with tuned parameters											 ##
###########################################################################################################################


class model():

    def __init__(self,data,X,y):

        ''' initialising the varibales
            X = List of features
            y = Target varibale
            data = pandas/h2o datafrae '''

        self.data = data
        self.X = X
        self.y = y       
    
    def train_valid_split(self, seed_val = 1000, train_size = 0.70, stratify = True):
        ''' Splitting the data into train and validation'''
        if stratify:
            test_size = 1- train_size
            stratsplit = self.data[self.y].stratified_split(test_frac= test_size, seed = seed_val)
            train = self.data[stratsplit=="train"]
            valid = self.data[stratsplit=="test"]
        else:
            train, valid= self.data.split_frame(ratios = train_size, seed = seed_val)
        return (train, valid)


    def foldsets(self,train,fold_num=3):
        # randomly assign fold numbers 0 through fold_num for each row in the column
        fold_numbers = train.kfold_column(n_folds = fold_num, seed = 1234)

        # rename the column "fold_numbers"
        fold_numbers.set_names(['fold_numbers'])

        # append the fold_numbers column to the train dataset
        train = train.cbind(fold_numbers)
        return (train)

    def build_tuned_model(self,model_name,train,valid,hyper_params,stopping_metric='auto',maximising_score = 'mcc'):

        '''	input:
            model name = 'gbm','drf'
            selected col = list of features
            target col = Target feature
            train = train data(h2o frame)
            valid = validation data (h2o frame)
            hyper_params = parameters need to be tuned for the specified model
            maximising_score = score which needs to be maximised

            output:
            best_param
        '''

        if model_name == 'gbm':
            print("Tuning the GBM model.....")
            grid_search_model = H2OGradientBoostingEstimator(
                stopping_rounds = 25,
                stopping_metric = stopping_metric,
                seed = 1234
            )

        elif model_name == 'drf':
            print("Tuning the Random Forest model....")
            grid_search_model = H2ORandomForestEstimator(
                # Stops fitting new trees when 10-tree rolling average is within 0.00001
                stopping_rounds = 10,
                stopping_tolerance = 0.00001,
                stopping_metric = 'auto',
                score_each_iteration = True,
                balance_classes = True,
                seed = 7)

        # grid search:
        grid = H2OGridSearch(grid_search_model, hyper_params = hyper_params,
                                 grid_id='depth_grid_search',
                                 search_criteria={'strategy': "Cartesian"})
        #Train grid search:
        grid.train(x=self.X,
                   y=self.y,
                   training_frame=train,
                   validation_frame=valid)
        
        # sorting the grid results based on the maximising score:
        grid_sorted = grid.get_grid(sort_by=maximising_score,decreasing=True)
        
        # the model with the best parameters:
        best_tuned_model = grid_sorted.models[0]
        return(best_tuned_model)


    def build_model(self,model_name,train,valid,fold_num=3):
        '''
        Build GBM | Randomforest model with the default parameters
        '''
        if model_name == 'gbm':
            print("Building the GBM model with the default Parameters....")
            classifier = H2OGradientBoostingEstimator(distribution="multinomial", seed = 1234)
            if fold_num is not None:
                train1 = self.foldsets(train,fold_num)
                classifier.train(x=self.X, y=self.y, training_frame=train1, validation_frame = valid, fold_column='fold_numbers')
            else:
                classifier.train(x=self.X, y=self.y, training_frame=train, validation_frame = valid)
    
        if model_name =='drf':
            print("Building the Randomforest model with the default parameters....")
            classifier = H2ORandomForestEstimator(distribution="multinomial", seed = 1234)
            if fold_num is not None:
                train = self.foldsets(train,fold_num)
                classifier.train(x=self.X, y=self.y, training_frame=train, validation_frame = valid,fold_column='fold_numbers')
            else:
                classifier.train(x=self.X, y=self.y, training_frame=train, validation_frame = valid)
        return (classifier)


    def download_pojo(self,model,path):
        '''
        Downloading the pojo and the JAR file
        '''
        try:
            h2o.download_pojo(model, path = path, get_jar = True)
            print("\t************Model Saved successfully******************")
        except Exception as e:
            logger.error('Failed to save the model: '+ str(e))
            
    def predict_test(self,model,test):
        '''
        Prediction of the test data
        '''
        pred = model.predict(test[self.X])
        print("Prediction for the test Data is done Successfully!!!")
        return(pred)

####################################################################################################################################
##                                             END - MODEL BUILDING MODULE                                                        ##
####################################################################################################################################
