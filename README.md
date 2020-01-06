# ML-Flow ![](https://img.shields.io/badge/Haribaskar-Dhanabalan-brightgreen.svg?colorB=#ADFF2F)

MLFlow is the generalized package which is developed using the python and the H2o framework. The purpose of this function is to store the models meta data ie., for both classification and regression model. 

---
Preperation of model meta data:
---
1. `data` = pandas dataframe or H2o dataframe based on which the below flags are set,
    h2oframe, pdframe
2. `split` = If the data is passed then based on this split - it divides the data in train and valid
        If train & valid is passed direclty then this can be ignored
3. `model_uuid` = unique id for user reference (If this is none an Unique ID will be generated automatically)
4. `model_type` = classification/ regression
5. `model version` = (by default Version1 is passed)
6. `model` = actual model data (as of now H2o model)
7. `features` = list of features (X)
8. `target` = target variable
9. `hyperparams_tuning` = if the model is tuned based on grid search. Then list of tuned parameters needs to be passed
10. `maximising metrics` = for train/valid metrics - it generated the metrics details for different sets of threshold, So based on this maximising metrics the threshold and corresponding metrics are stores.
11. `plots_show` = To display all the metrics graphs for the passed model.  For both train and validation data.


The use case I have solved in this module is to get the model meta data. 

<p align="center">
    <img src="https://github.com/hari2594/ML-Flow/blob/master/Template.PNG" width="250" />
</p>

**The modules handled in this pacakges are:**
1. Does the model training and the predicition, based on the inputs passed. ( The inputs passed can be either a pandas or H2o framework
   correspondingly the respective flags should be set.
2. It also handles the hyper parameter tuning to get the best parameters.
3. Evaluation fo the model (Please refer to my other repo [Model Evaluation](https://github.com/hari2594/Model-Evaluation-Comparison)
4. Getting the best threshold for Classification models by maximising a metrics.
5. And then creates a overall information about the model - which is the model meta data in the json format

---
## Sample Execution Output:

```python
getting the actual and predicted values...
gbm prediction progress: |████████████████████████████████████████████████| 100%
gbm prediction progress: |████████████████████████████████████████████████| 100%
Getting the metrics details for the train data...
Getting the metrics details for the validation data...
Model UUID is not given - random UUID is created
Model Tag is not given - NULL is passed
Model version is not given - 'Version-1' is passed
Model type is not given - None is passed
hyperparams_tuning is not given - None is passed
Seed value is not passed in the Model - seed = 0 is passed as default
No cross-validation metrics summary for this model
CV is not performed in the Model - NULL is passed
  .........
```

```
Sample meta data:
{`Model_UUID`: `17b68386-d49a-11e9-8bff-3c2c30d1ea45`,
 `Model_Tag`: `None`,
 `Model_Version`: `Version-1`,
 `Model_Type`: `None`,
 `Rundate`: datetime.datetime(2019, 9, 11, 13, 43, 9, 944992),
 `Trainining_Start_Time`: datetime.datetime(2019, 9, 11, 13, 43, 9, 944992),
 `Traning_End_Time`: datetime.datetime(2019, 9, 11, 13, 43, 9, 944992),
 `Total_Training_Time`: None,
 `Trained_By`: `Haribaskar.d`,
 `Input_Features`: "[`X1`, `X2`, `X3`, `X4`]",
 `Hyperparams_Tuning`: `None`,
 `Hyperparams`: "",
 `Seed_Value`: 0,
 `Training_Data_Reference`: None,
 `Test_Data_Reference`: None,
 `Model_Path`: None,
 `Model`: `NULL`,
 `Feature_Importance`: "[(`X1`, 2339.82958984375, 1.0, 0.7399357915895), (`X2`, 449.2460632324219, 0.19199947944175788, 0.14206728680550904), (`X3`, 334.8724670410156, 0.14311831446809675, 0.10589836330690615), (`X4`, 38.25813674926758, 0.016350821835628804, 0.01209855829808488)]",
 `Feature_Distribution`: "{`X1`: {`mean`: 0.06662887410836711, `std`: 0.9717070855781386, `missing%`: 0.0}, `X2`: {`mean`: 0.0823587388025514, `std`: 0.9579708217449495, `missing%`: 13.906272740503567}, `X3`: {`mean`: 0.07412937544293083, `std`: 0.9615146723706843, `missing%`: 28.882258768738172}, `X4`: {`mean`: 0.44178431087177994, `std`: 0.7264934356403574, `missing%`: 0.0}}",
 `Model_Accuracy_Metrics_Cv`: `None`,
 `Model_Accuracy_Metrics_Training`: "[{`TP`: 7662.0, `FP`: 651.0, `FN`: 1022.0, `TN`: 1659.0, `Accuracy`: 0.8478260869565217, `Precision0`: 0.618798955613577, `Precision1`: 0.9216889209671598, `recall0`: 0.7181818181818181, `recall1`: 0.8823122984799632, `f1`: 0.9015708654468435, `mcc`: 0.5697014920234555, `roc_auc`: 0.8002470583308907}]",
 `Model_Accuracy_Metrics_Validation`: "[{`TP`: 1788.0, `FP`: 153.0, `FN`: 383.0, `TN`: 424.0, `Accuracy`: 0.8049490538573508, `Precision0`: 0.5254027261462205, `Precision1`: 0.9211746522411128, `recall0`: 0.7348353552859619, `recall1`: 0.8235836020267158, `f1`: 0.8696498054474707, `mcc`: 0.4993768857270868, `roc_auc`: 0.7792094786563388}]",
 `Model_Parameters`: None,
 `Model_Summary_Stats`: None,
 `Model_Plots`: None,
 `Training_Count`: 10994,
 `Validation_Count`: 2748,
 `Total_Count`: 13742,
 `Best_Threshold`: 0.67,
 `Best_Metric`: `mcc`,
 `Threshold_Used`: 0.67,
 `Created_Date`: datetime.datetime(2019, 9, 11, 13, 43, 10, 188889),
 `Created_By`: `Haribaskar.d`,
 `Modified_Date`: datetime.datetime(2019, 9, 11, 13, 43, 10, 188889),
 `Modified_By`: None}
 ```
---
## Contributing

Patches are welcome, preferably as pull requests.

---



