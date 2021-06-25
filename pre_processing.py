import pandas as pd
import os
import numpy as np
from pathlib import Path
from tadpole_algorithms.models import EMCEB
from tadpole_algorithms.models import BenchmarkLastVisit
from tadpole_algorithms.models import BenchmarkSVM
from tadpole_algorithms.preprocessing.split import split_for_predict_combination
#from tadpole_algorithms.preprocessing.split import split_train_val_NN
from tadpole_algorithms.preprocessing.split import split_train_50_50
import os
from pathlib import Path


def get_predictions_EMCEB(train_df, test_df, true_labels, train, test, model):

    ### Maximum number of datapoints per RID
    max_per_RID = test_df.groupby("RID").size().max()
    ### Define predictions dataframe
    total_predictions = pd.DataFrame()

    ## make copy of train_df and test_df
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()

    #### Predict
    ### predict one row per RID every iteration and append predictions of each iteration
    for i in range(max_per_RID):

        ### Only train model once and set data for training once
        ### but set test_df every iteration so that right test_df is predicted
        
        if i == 0:

            # Define model
            #model = EMCEB()
            model.set_data(train_df, test_df, train, test)
            # Train 
            model.train()

            ## Reset train_df and test_df
            test_df = test_df_copy
            train_df = train_df_copy

            ### Add column index to dataframe
            test_df["index"] = test_df.index

        ### select first row
        first_row = test_df.groupby('RID').head(1)
        
        ## here it is about setting test_df to first row and not about setting train_df
        model.set_data(train_df, first_row, train, test)
        
        first_row.index = first_row["index"]
        ## drop first row so that in next iteration, de next row becomes first_row
        test_df = test_df.drop(first_row.index)

        predictions = model.predict()

        ########### Search the right prediction from 10 years prediction
        true_labels["index"] = true_labels.index
        first_row_true_label = true_labels.groupby('RID').head(1)

        nrSubj = first_row_true_label.shape[0]
        predictions['Forecast Date'] = pd.to_datetime(predictions['Forecast Date'])
        output = pd.DataFrame()

        for s in range(nrSubj):
            currSubjMask = first_row_true_label['RID'].iloc[s] == predictions['RID']
            currSubjData = predictions[currSubjMask]

            timeDiffsScanCog = [pd.to_datetime(first_row_true_label['EXAMDATE']).iloc[s] - d for d in currSubjData['Forecast Date']]

            indexMin = np.argsort(np.abs(timeDiffsScanCog))[0]

            currSubjData = currSubjData.iloc[[indexMin]]

            output = output.append(currSubjData)
            print(s)

        total_predictions = total_predictions.append(output)

        ## drop first row so that in next iteration, de next row becomes first_row
        true_labels = true_labels.drop(first_row_true_label.index)

        ## reset train_df
        train_df = train_df.copy()

        total_predictions = total_predictions.sort_values(by = ["RID", "Forecast Date"])

    return total_predictions

def get_predictions_benchmarks(train_df, test_df, true_labels, model):

    ### Maximum number of datapoints per RID
    max_per_RID = test_df.groupby("RID").size().max()
    ### Define predictions dataframe
    total_predictions = pd.DataFrame()
    test_df = test_df.fillna(0)

    #### Predict
    ### predict one row per RID every iteration and append predictions of each iteration
    for i in range(max_per_RID):

        ### Only train model once and set data for training once
        ### but set test_df every iteration so that right test_df is predicted

        
        if i == 0:
            ### Train 
            model.train(train_df)

            ### Add column index to dataframe
            test_df["index"] = test_df.index

        ### select first row
        first_row = test_df.groupby('RID').head(1)
        
        first_row.index = first_row["index"]
        ## drop first row so that in next iteration, de next row becomes first_row
        test_df = test_df.drop(first_row.index)

        predictions = model.predict(first_row)
        predictions = predictions.rename(columns={'month': 'Forecast Month'})

        # Save predictions files to use for other algorithms
        data_path_predictions = Path("jupyter/data/Prediction_format_combination_data/predictions{0}.xlsx".format(i+1))
        predictions.to_excel(data_path_predictions)

        ########### Search the right prediction from 10 years prediction
        true_labels["index"] = true_labels.index
        first_row_true_label = true_labels.groupby('RID').head(1)

        nrSubj = first_row_true_label.shape[0]
        predictions['Forecast Date'] = pd.to_datetime(predictions['Forecast Date'])
        output = pd.DataFrame()

        for s in range(nrSubj):
            currSubjMask = first_row_true_label['RID'].iloc[s] == predictions['RID']
            currSubjData = predictions[currSubjMask]

            timeDiffsScanCog = [pd.to_datetime(first_row_true_label['EXAMDATE']).iloc[s] - d for d in currSubjData['Forecast Date']]

            indexMin = np.argsort(np.abs(timeDiffsScanCog))[0]

            currSubjData = currSubjData.iloc[[indexMin]]

            output = output.append(currSubjData)
            print(s)

        total_predictions = total_predictions.append(output)
        print(len(total_predictions))

        ## drop first row so that in next iteration, de next row becomes first_row
        true_labels = true_labels.drop(first_row_true_label.index)

        total_predictions = total_predictions.sort_values(by = ["RID", "Forecast Date"])

    return total_predictions

def get_true_labels(dataPathTest):

    # Read data
    dataSetTest = pd.read_csv(dataPathTest)

    # Get converters
    msk = dataSetTest['Diagnosis']!= dataSetTest['DX_LastVisitADNI2']
    converters_test = msk*1
    converters_test = pd.DataFrame(converters_test, columns = ['Converters'])
    converters_test = converters_test['Converters'].astype(float)
    converters_test = pd.DataFrame(converters_test, columns = ['Converters'])



    # todo: Check mapping
    if 'Diagnosis' not in dataSetTest.columns:
        dataSetTest = dataSetTest.replace({'DXCHANGE': {
            1: 'CN',
            2: 'MCI',
            3: 'AD',
            4: 'MCI',
            5: 'AD',
            6: 'AD',
            7: 'CN',
            8: 'MCI',
            9: 'CN'
        }})
        dataSetTest = dataSetTest.replace({'DXCHANGE': {
            'CN': 0,
            'MCI': 1,
            'AD': 2,
        }})
        dataSetTest = dataSetTest.rename(columns={"DXCHANGE": "Diagnosis"})
    else:
        dataSetTest = dataSetTest.replace({'Diagnosis': {
            'CN': 0,
            'MCI': 1,
            'AD': 2,
        }})       

    dataSetTest = dataSetTest.rename(columns={'Ventricles': 'Ventricles_ICV'})
    true_labels_test = dataSetTest[["RID",'Diagnosis', 'ADAS13', "Ventricles_ICV"]]

    return true_labels_test, converters_test
