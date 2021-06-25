import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from tadpole_algorithms.models import BenchmarkLastVisit
from tadpole_algorithms.models import EMCEB
from tadpole_algorithms.models import BenchmarkSVM

from tadpole_algorithms.preprocessing.split import split_test_train_tadpole
from tadpole_algorithms.preprocessing.split import split_train_50_50
from tadpole_algorithms.tests.neuralnet.eval import evaluate_forecastt

def search_right_date(dfD4, predictions_test):
    
    nrSubj = dfD4.shape[0]
    output = pd.DataFrame()
    total_predictions = pd.DataFrame()
    for s in range(nrSubj):
        currSubjMask = dfD4['RID'].iloc[s] == predictions_test['RID']
        currSubjData = predictions_test[currSubjMask]

        timeDiffsScanCog = [pd.to_datetime(dfD4['CognitiveAssessmentDate']).iloc[s] - d for d in pd.to_datetime(currSubjData['Forecast Date'])]

        indexMin = np.argsort(np.abs(timeDiffsScanCog))[0]
        currSubjData = currSubjData.iloc[[indexMin]]

        output = output.append(currSubjData)
        print(s)

    total_predictions = total_predictions.append(output)
    print(len(total_predictions))

    return total_predictions

def predict_individual_models_and_bootstrap(data_df_train_test, data_df_eval, model, model_name, SCENARIO_2):
    if SCENARIO_2 == True:
        _, _, eval_df = split_test_train_tadpole(data_df_train_test, data_df_eval)

        # Use same train_df as ensembles use
        train_df = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/train_df_for_split_RIDS.csv')
        train_df = train_df.drop(columns = 'Unnamed: 0')
        test_df = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/test_df_for_split_RIDS.csv')
        test_df = test_df.drop(columns = 'Unnamed: 0')

    else:
        # Split data in test, train and evaluation data
        train_df, test_df, eval_df = split_test_train_tadpole(data_df_train_test, data_df_eval)



    if model_name == 'BenchmarkLastVisit':
        
        test_df = test_df.fillna(0)

        model.train(train_df)

        # Predict forecast on the test set
        forecast_df_d2_blv = model.predict(test_df)
        predictions_test_blv = search_right_date(eval_df, forecast_df_d2_blv)

        # Evaluate forecast
        _, _, results, TrueFalse = evaluate_forecastt(eval_df, predictions_test_blv)

        if SCENARIO_2 == True:
            # results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind/BLV_results_100_btsr2.csv')
            # Forecast to be used for testing ensembles
            # predictions_test_blv.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_blv_50p.csv')
            TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/blv_50p.csv')

        else:
            results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind_100p/BLV_results_100_btsr.csv')
            
            # Forecast to be used for testing ensembles
            predictions_test_blv.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_blv_100p.csv')
            TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/blv_100p.csv')


    elif model_name == 'BenchmarkSVM':

        test_df = test_df.fillna(0)

        model.train(train_df)

        # Predict forecast on the test set
        forecast_df_d2_bsvm = model.predict(test_df)
        predictions_test_bsvm = search_right_date(eval_df, forecast_df_d2_bsvm)
        
        # Evaluate forecast
        _, _, results, TrueFalse = evaluate_forecastt(eval_df, predictions_test_bsvm)

        if SCENARIO_2 == True:
            # results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind/BSVM_results_100_btsr2.csv')
            # Forecast to be used for testing ensembles
            # predictions_test_bsvm.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_bsvm_50p.csv')
            TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/bsvm_50p.csv')
        else:
            results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind_100p/BSVM_results_100_btsr.csv')
            # Forecast to be used for testing ensembles
            predictions_test_bsvm.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_bsvm_100p.csv')
            TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/bsvm100p.csv')

    elif model_name == 'EMCEB':

        train = "d1d2"
        test = "d1d2"

        # Define model
        model = EMCEB()

        # Preprocess and set data 
        model.set_data(train_df, test_df, train, test)

        # Train model
        # Note to self: number of bootstraps set to 1 for computation speed. Should be 100 to compute CIs.
        model.train()

        # Predict forecast on the test set
        forecast_df_d2_emceb = model.predict()

        predictions_test_emceb = search_right_date(eval_df, forecast_df_d2_emceb)

        _, _, results, TrueFalse = evaluate_forecastt(eval_df, predictions_test_emceb)

        if SCENARIO_2 == True:
            # results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind/EMCEB_results_100_btsr2.csv')
            # Forecast to be used for testing ensembles
            # predictions_test_emceb.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_emceb_50p.csv')
            TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/emceb_50p.csv')

        else:
            results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind_100p/EMCEB_results_100_btsr.csv')
            predictions_test_emceb.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_emceb_100p.csv')
            TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/emceb_100p.csv')

    elif model_name == 'EMC1':
        # Get forecast on 100 % of D2
        forecast_df_d2_emc1 = pd.read_csv('tadpole_algorithms/tests/Outputs/emc1/forecast_df_d2.csv')
        predictions_test_emc1 = search_right_date(eval_df, forecast_df_d2_emc1)

        _, _, results, TrueFalse = evaluate_forecastt(eval_df, predictions_test_emc1)
        results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind_100p/EMC1_results_100_btsr.csv')
        predictions_test_emc1.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_emc1_100p.csv')
        TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/emc1_100p.csv')

    elif model_name == 'BORREGOSTECMTY':
        # Get forecast on 100 % of D2
        forecast_df_d2_mexico = pd.read_csv('tadpole_algorithms/tests/Outputs/BORREGOSTECMTY/forecast_df_d2.csv')
        predictions_test_mexico = search_right_date(eval_df, forecast_df_d2_mexico)

        _, _, results, TrueFalse = evaluate_forecastt(eval_df, predictions_test_mexico)
        results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_ind_100p/Mexico_results_100_btsr.csv')
        predictions_test_mexico.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Intermediate_forecast/predictions_test_btmty_100p.csv')
        TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/btmty_100p.csv')


    return  









