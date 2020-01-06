

##########################################################################################################################
##  Importing the necessary libraries:                                                                                  ##
##########################################################################################################################

import os
import uuid
from datetime import datetime
import time
import os
import evaluation_module
from distribution_details import distribution_check_detail
from datetime import datetime
from time import gmtime, strftime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sqlalchemy
import urllib
import h2o


##########################################################################################################################
## Model's Meta Data Prepaeration:                                                                                      ##
##########################################################################################################################

class model_meta_data():
    '''
    Preperation of model meta data:
    -------------------------------
    data = pandas dataframe or H2o dataframe based on which the below flags are set,
        h2oframe, pdframe
    split = If the data is passed then based on this split - it divides the data in train and valid
            If train & valid is passed direclty then this can be ignored
    model_uuid = unique id for user reference (If this is none an Unique ID will be generated automatically)
    model_type = classification/ regression
    model version = (by default Version1 is passed)
    model = actual model data (as of now H2o model)
    hyperparams_tuning = if the model is tuned based on grid search. Then list of tuned parameters needs to be passed
    maximising metrics = for train/valid metrics - it generated the metrics details for different sets of threshold, So based on this maximising metrics the threshold and corresponding metrics are stores.
    '''
    
    def __init__(self,data = None, split = 0.80, train=None, valid = None, features = None, target = None,
                 model_uuid = None, model_tag = None, model_version = None, model_type = None,
                 model = None, hyperparams_tuning = None, maximising_metrics = 'f1', 
                 h2oframe = True, pdframe = False):
        
        self.data = data
        self.features = features
        self.target = target
        self.split = split
        self.train = train
        self.valid = valid
        self.model_uuid = model_uuid
        self.model_tag = model_tag
        self.model_version = model_version
        self.model_type = model_type
        self.model = model
        self.hyperparams_tuning = hyperparams_tuning
        self.maximising_metrics = maximising_metrics
        self.h2oframe = h2oframe
        self.pdframe = pdframe
        
    def model_UUID(self):
        if self.model_uuid is not None:
            return(self.model_uuid)
        else:
            print("Model UUID is not given - random UUID is created")
            return(uuid.uuid1())
        
    def model_TAG(self):
        if self.model_tag is not None:
            return(self.model_tag)
        else:
            print("Model Tag is not given - NULL is passed")
            return(None)
        
    def model_VERSION(self):
        if self.model_version is not None:
            return(self.model_version)
        else:
            print("Model version is not given - 'Version-1' is passed")
            return('Version-1')
        
    def model_TYPE(self):
        if self.model_type is not None:
            return(self.model_type)
        else:
            print("Model type is not given - None is passed")
            return(None)
        
    def model_STARTTIME(self):
        if self.model is not None:
            return(self.model.start_time)
        else:
            print("Model is not given - model_STARTTIME = None is passed")
            return(None)

    def model_ENDTIME(self):
        if self.model is not None:
            return(self.model.end_time)
        else:
            print("Model is not given - model_ENDTIME = None is passed")
            return(None)
        
    def model_TOTALTIME(self):
        if self.model is not None:
            return((self.model.end_time - self.model.start_time)/60) #converting it into minutes
        else:
            print("Model is not given - model_TOTALTIME = None is passed")
            return(None)
    
    def model_FEATURE_IMPORTANCE(self):
        if self.model is not None:
            return(self.model.varimp())
        else:
            print("Model is not given - model_FEATURE_IMPORTANCE = None is passed")
            return(None)
    
    def model_HYPERPARAMS(self):
        if self.model is not None:
            return(self.model.params)
        else:
            print("Model is not given - model_HYPERPARAMS = None is passed")
            return(None)
            
    def model_SEED(self):
        if self.model is not None:
            if self.model.seed is not None:
                return(self.model.seed)
            else:
                print("Seed value is not passed in the Model - seed = 0 is passed as default")
                return(0)
        else:
            print("Model is not given - model_SEED = None is passed")
            return(None)
        
    def model_INPUT_FEATURES(self):
        if self.features is not None:
            return(self.features)
        elif self.model is not None:
            return([self.model.varimp()[i][0] for i in range(len(self.model.varimp()))]) #converting it inot minutes
        else:
            print("Features is not given - input_features = None is passed")
            return(None)
        
    def model_TRAINED_BY(self):
        ''' Returns the current login user-name'''
        return(os.getlogin())
    
    def model_HYPERPARAM_TUNING(self):
        if self.hyperparams_tuning is not None:
            return(self.hyperparams_tuning) 
        else:
            print("hyperparams_tuning is not given - None is passed")
            return(None)
        
    def model_TRAIN_COUNT(self):
        if self.train is not None:
            return(len(self.train)) 
        elif ((self.data is not None) and (self.split is not None)):
            return(round(len(self.data)*self.split))
        else:
            print("Data | train | valid - all are empty - count=0 is passed")
            return(None)
        
    def model_VALID_COUNT(self):
        if self.valid is not None:
            return(len(self.valid)) 
        elif ((self.data is not None) and (self.split is not None)):
            return(round(len(self.data)*(1-self.split)))
        else:
            print("Data | train | valid - all are empty - count=0 is passed")
            return(None)
        
    def model_TOTAL_COUNT(self):
        if ((self.valid is not None) and (self.train is not None)):
            return(len(self.valid) + len(self.train)) 
        elif ((self.data is not None) and (self.split is not None)):
            return(len(self.data))
        else:
            print("Data | train | valid - all are empty - count=0 is passed")
            return(None)
        
    def model_CV_METRICS(self):
        try:
            if self.model is not None:
                return(self.model.cross_validation_metrics_summary().as_data_frame().set_index(['']).to_dict()) 
            else:
                print("Model is not given - model_SEED = None is passed")
                return(None)
        except:
            print("CV is not performed in the Model - NULL is passed")
            return(None)
            
    def model_DISTRIBUTION_DETAILS(self):
        if self.data is not None:
            if self.h2oframe:
                _data = self.data.as_data_frame()
            else:
                _data = self.data
            # getting the distibution details:
            return(distribution_check_detail(data=_data,features=self.features))
        elif ((self.valid is not None) and (self.train is not None)):
            # concatinating the validation and the training data:
            full_data = pd.concat([self.valid,self.train],axis=0)
            # getting the distribution details
            return(distribution_check_detail(data=full_data,features=self.features))


#############################################################################################################################
## Class for getting the Train and Validation metrics 																	   ##
#############################################################################################################################

class model_train_valid_metrics():
    '''
    data = pandas dataframe or H2o dataframe based on which the below flags are set,
        h2oframe, pdframe
    train_size = If the data is passed then based on this split - it divides the data in train and valid
            If train & valid is passed direclty then this can be ignored
    model = actual model data (as of now H2o model)
    cat_features = list of cateforical features
    features = list of features (X)
    target = target variable
    plots_show = To display all the metrics graphs for the passed model.  For both train and validation data.
    '''
    
    def __init__(self, data=None, seed_val = 1000, train_size=0.80, stratify = True, valid=None, train=None, model=None, 
                 cat_features = None, features = None,
                 target=None, h2oframe = False, pdframe = True, plots_show = False):
        
        self.data = data
        self.valid = valid
        self.train = train
        self.model = model
        self.features = features
        self.target = target
        self.h2oframe = h2oframe
        self.pdframe = pdframe
        self.seed_val = seed_val
        self.train_size = train_size
        self.plots_show = plots_show
        self.stratify = stratify
        self.cat_features = cat_features
        
    def convert_h20df(self,data=None):
        '''
        Converting the pandas dataframe to h2o frame
        '''
        data = h2o.H2OFrame(data)
        for feat in self.cat_features:
            data[feat] = data[feat].asfactor()
        return data
    
    def get_actual_pred(self, data=None):
        '''
        Getting the Actual and the predicted values.
        Note: The data passed here should be h2o dataframe
        '''
        pred_val = self.model.predict(data).as_data_frame()
        df=data.as_data_frame()
        df['p1'] = pred_val['p1']
        df['p0'] = pred_val['p0']
        df['predict'] = pred_val['predict']
        pred_prob = list(df['p1'].values)
        actual = list(df[self.target].values)
        return(actual,pred_prob)
    
    def train_valid_split(self, data = None):
        ''' Splitting the data into train and validation'''
        if self.stratify:
            test_size = 1- self.train_size
            stratsplit = data[self.target].stratified_split(test_frac= test_size, seed = self.seed_val)
            train = data[stratsplit=="train"]
            valid = data[stratsplit=="test"]
        else:
            train, valid= data.split_frame(ratios = self.train_size, seed = self.seed_val)
        return (train, valid)
    
    def get_metric(self, maximising_metric = 'f1'):
        
        # if the data is passed as a pandas dataframe - It is converted into H20 frame
        if self.pdframe:
            print("Converting Pandas dataframe to H2OFrame...")
            if self.data is not None:
                print("Splitting the data into train & validation data...")
                self.data = self.convert_h20df(data=self.data)
                self.train, self.valid = self.train_valid_split(data = self.data)
            elif self.train is not None:
                self.train = self.convert_h20df(data=self.train)
            elif self.valid is not None:
                self.valid = self.convert_h20df(data=self.valid)
        elif self.h2oframe:
            if self.data is not None:
                self.train, self.valid = self.train_valid_split(data = self.data)
                        
        # getting the actual and the pred value for the train and validation data:
        print("getting the actual and predicted values...")
        train_actual, train_pred_prob = self.get_actual_pred(data=self.train)
        valid_actual, valid_pred_prob = self.get_actual_pred(data=self.valid)
        
        # getting the metrics and other details - by maximising a particular metrics:
        
        # TRAIN:
        print("Getting the metrics details for the train data...")
        evalu_train = evaluation_module.ModelEvaluation(actual = train_actual, pred = train_pred_prob, 
                                                        threshold = np.arange(0,1,0.01),
                                                        maximising_metrics=maximising_metric)
        train_metric_db, train_decile_db, train_best_threshold, train_maximising_metrics = evalu_train.evaluate(plots_show = self.plots_show)
        train_metric_db = train_metric_db[train_metric_db['Threshold'] == train_best_threshold].reset_index(drop=True)
        # converting the results to dictionary:
        df= train_metric_db[:1][train_metric_db.columns[3:15]]
        df.set_index('TP')[list(df.columns)[1:]].to_dict()
        train_metric_db = df.to_dict('r')
        
        # VALIDATION:
        print("Getting the metrics details for the validation data...")
        evalu_valid = evaluation_module.ModelEvaluation(actual = valid_actual, pred = valid_pred_prob, 
                                                        threshold = np.arange(0,1,0.01),
                                                        maximising_metrics=maximising_metric)
        valid_metric_db, valid_decile_db, valid_best_threshold, valid_maximising_metrics = evalu_valid.evaluate(plots_show = self.plots_show)
        valid_metric_db = valid_metric_db[valid_metric_db['Threshold'] == valid_best_threshold].reset_index(drop=True)
        # converting theresults to dictionary:
        df= valid_metric_db[:1][valid_metric_db.columns[3:15]]
        df.set_index('TP')[list(df.columns)[1:]].to_dict()
        valid_metric_db = df.to_dict('r')
        # retuning the metric and other detials for train and validation:
        return(train_metric_db, train_decile_db, train_best_threshold, train_maximising_metrics,valid_metric_db, 
               valid_decile_db, valid_best_threshold, valid_maximising_metrics)


#############################################################################################################################
## MLFlow - To get the Model Meta data and to store the meta data in DB      											   ##
#############################################################################################################################

class MLflow():

    '''
    Preperation of model meta data:
    -------------------------------
    data = pandas dataframe or H2o dataframe based on which the below flags are set,
        h2oframe, pdframe
    split = If the data is passed then based on this split - it divides the data in train and valid
            If train & valid is passed direclty then this can be ignored
    model_uuid = unique id for user reference (If this is none an Unique ID will be generated automatically)
    model_type = classification/ regression
    model version = (by default Version1 is passed)
    model = actual model data (as of now H2o model)
    features = list of features (X)
    target = target variable
    hyperparams_tuning = if the model is tuned based on grid search. Then list of tuned parameters needs to be passed
    maximising metrics = for train/valid metrics - it generated the metrics details for different sets of threshold, So based on this maximising metrics the threshold and corresponding metrics are stores.
    plots_show = To display all the metrics graphs for the passed model.  For both train and validation data.
    '''
    
    # declaring a global variable
    global metadata
    metadata = dict()
    
    
    def __init__(self,data=None, split=0.80, train_data=None, valid_data=None, features= None, target = None, model_uuid = None,
                 model_tag = None, model_version = None, model_type = None, model = None, hyperparams_tuning = None,
                 maximising_metrics = 'f1',plots_show = False,h2oframe = False, pdframe = True):
        
        self.data = data
        self.train_data = train_data
        self.valid_data = valid_data
        self.split = split
        self.features = features
        self.target = target
        self.model_uuid = model_uuid
        self.model_tag = model_tag
        self.model_version = model_version
        self.model_type = model_type
        self.model = model
        self.hyperparams_tuning = hyperparams_tuning
        self.maximising_metrics = maximising_metrics
        self.plots_show = plots_show
        self.h2oframe = h2oframe
        self.pdframe = pdframe
        self._train_metric_db = None
        self._train_best_threshold=None
        self._train_maximising_metrics = None
        self._valid_metric_db = None
        self._valid_best_threshold=None
        self._valid_maximising_metrics = None
        
    def train_val_metric(self):
        '''
        Getting the metric details for the train and validationd data
        '''
        mod_tv = model_train_valid_metrics(data=self.data, seed_val = 1000, 
                                        train_size=self.split, valid=self.valid_data, train=self.train_data, 
                                        model=self.model, cat_features = [self.target], features = self.features,
                                        target=self.target, stratify = True, h2oframe = self.h2oframe, pdframe = self.pdframe, 
                                        plots_show = self.plots_show)
        self._train_metric_db, _train_decile_db, self._train_best_threshold,self._train_maximising_metrics,self._valid_metric_db,_valid_decile_db, self._valid_best_threshold,self._valid_maximising_metrics = mod_tv.get_metric(maximising_metric = self.maximising_metrics)
        
    def train(self):
        '''
        Preperation of the model meta data
        '''

        # initialising the meta data class:
        md = model_meta_data(data = self.data, split = 0.80, train=self.train_data, valid = self.valid_data, 
                       features = self.features, target = None,model_uuid = self.model_uuid, model_tag = self.model_tag, 
                       model_version = self.model_version, model_type = self.model_type,
                       model = self.model, hyperparams_tuning = self.hyperparams_tuning,
                       maximising_metrics = self.maximising_metrics,
                       h2oframe = self.h2oframe, pdframe = self.pdframe)
        
        # Running the train_valid_metric function:
        self.train_val_metric()
        
        # meta data preperation:
        metadata['Model_UUID'] = str(md.model_UUID())
        metadata['Model_Tag'] = str(md.model_TAG())
        metadata['Model_Version'] = str(md.model_VERSION())
        metadata['Model_Type'] = str(md.model_TYPE())
        metadata['Rundate'] = datetime.utcnow()
        metadata['Trainining_Start_Time'] = datetime.utcnow()
        metadata['Traning_End_Time'] = datetime.utcnow()
        metadata['Total_Training_Time'] = None
        metadata['Trained_By'] = str(md.model_TRAINED_BY())
        metadata['Input_Features'] = str(md.model_INPUT_FEATURES())
        metadata['Hyperparams_Tuning'] = str(md.model_HYPERPARAM_TUNING())
        metadata['Hyperparams'] = str(md.model_HYPERPARAMS())
        metadata['Seed_Value'] = int(md.model_SEED())
        metadata['Training_Data_Reference'] = None
        metadata['Test_Data_Reference'] = None
        metadata['Model_Path'] = None
        metadata['Model'] = "NULL"
        metadata['Feature_Importance'] = str(md.model_FEATURE_IMPORTANCE())
        metadata['Feature_Distribution'] = str(md.model_DISTRIBUTION_DETAILS())
        metadata['Model_Accuracy_Metrics_Cv'] = str(md.model_CV_METRICS())
        metadata['Model_Accuracy_Metrics_Training'] = str(self._train_metric_db)
        metadata['Model_Accuracy_Metrics_Validation'] = str(self._valid_metric_db)
        metadata['Model_Parameters'] = None
        metadata['Model_Summary_Stats'] = None
        metadata['Model_Plots'] = None
        metadata['Training_Count'] = int(md.model_TRAIN_COUNT())
        metadata['Validation_Count'] = int(md.model_VALID_COUNT())
        metadata['Total_Count'] = int(md.model_TOTAL_COUNT())
        metadata['Best_Threshold'] = self._train_best_threshold
        metadata['Best_Metric'] = self._train_maximising_metrics
        metadata['Threshold_Used'] = self._train_best_threshold
        metadata['Created_Date'] = datetime.utcnow()
        metadata['Created_By'] = str(md.model_TRAINED_BY())
        metadata['Modified_Date'] = datetime.utcnow()
        metadata['Modified_By'] = None

        return(metadata)
        
    def save(self):
        '''
        To upload the model meta data in DB - with the Model UUID as the primary key.
        '''
        try:
            data_upload = pd.DataFrame([metadata])

            # db configuration details:
            input_mssql_config =  {'driver': '{SQL Server}' ,
                   'host': "***" ,
                   'database': "****",
                   'user': "****" ,
                   'password': "*****"}
            # default odbc format:
            odbc_string = 'DRIVER={driver};SERVER={server};DATABASE={db};UID={user_id};PWD={password};ansi=True'

            # configuration details formatting:
            input_odbc_string = odbc_string.format(driver=input_mssql_config['driver'],
                                                   server=input_mssql_config['host'],
                                                   db=input_mssql_config['database'],
                                                   user_id=input_mssql_config['user'],
                                                   password=input_mssql_config['password'])

            model_store_engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % urllib.parse.quote_plus(input_odbc_string))
            data_upload.to_sql('ml_model_store', schema='Predict', con=model_store_engine, if_exists='append', index=False)
            print("Data Loaded successfully: Model_UUID = ",metadata['Model_UUID'])
        except:
            print("Data Loading failed")
    
    def get_data(self):
        '''
        It gets the models meta data.
        '''
        if len(metadata) > 0:
            return(metadata)
        else:
            self.train()
            return(metadata)