from pathlib import Path
import os
import numpy as np
import pandas as pd
#from tadpole_algorithms.preprocessing.split import split_train_val_NN
from tadpole_algorithms.preprocessing.split import split_for_predict_combination
from tadpole_algorithms.preprocessing.split import split_train_50_50
from tadpole_algorithms.tests.neuralnet.pre_processing import get_true_labels
from tadpole_algorithms.models import EMCEB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


###############
### Make sure to have obtained predictions for the all of the 3 algorithms blv, bsvm and emceb
### these can be obtained via the scripts: predict_for_NN_benchmark_lv, predict_for_NN_benchmark_svm, predict_for_NN_emceb
X_IND_TEST= pd.read_csv("tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/X_IND_TEST_features2.csv")
X_TEST= pd.read_csv("tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/X_TEST_features2.csv")
X_TRAIN = pd.read_csv("tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/X_TRAIN_features2.csv")
Y_TRAIN = pd.read_csv("tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/Y_TRAIN_features2.csv")

X_IND_TEST = X_IND_TEST.drop(columns = "Unnamed: 0")
X_TEST = X_TEST.drop(columns = "Unnamed: 0")
X_TRAIN = X_TRAIN.drop(columns = "Unnamed: 0")
Y_TRAIN = Y_TRAIN.drop(columns = {"Unnamed: 0", 'index'})

# Normalise
x = X_IND_TEST.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_IND_TEST = pd.DataFrame(x_scaled, columns=X_IND_TEST.columns)

x = X_IND_TEST.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_IND_TEST = pd.DataFrame(x_scaled, columns=X_IND_TEST.columns)

x = X_TEST.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_TEST = pd.DataFrame(x_scaled, columns=X_TEST.columns)


x = X_TEST.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_TEST = pd.DataFrame(x_scaled, columns=X_TEST.columns)

x = X_TRAIN.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_TRAIN = pd.DataFrame(x_scaled, columns=X_TRAIN.columns)

x = X_TRAIN.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X_TRAIN = pd.DataFrame(x_scaled, columns=X_TRAIN.columns)



data_path_emceb_train_val = Path('tadpole_algorithms/tests/Outputs/CombinationData/emceb/Random_split_train_val/NN_train_val_df_splittrain5050.csv')
data_path_emceb_test = Path('tadpole_algorithms/tests/Outputs/CombinationData/predictions_test_emceb.csv')

data_path_blv_train_val = Path('tadpole_algorithms/tests/Outputs/CombinationData/benchmark_last_visit/Random_split_train_val/NN_train_val_df_splittrain5050.csv')
data_path_blv_test = Path('tadpole_algorithms/tests/Outputs/CombinationData/predictions_test_blv.csv')

data_path_bsvm_train_val = Path('tadpole_algorithms/tests/Outputs/CombinationData/benchmark_svm/Random_split_train_val/NN_train_val_df_splittrain5050.csv')
data_path_bsvm_test = Path('tadpole_algorithms/tests/Outputs/CombinationData/predictions_test_bsvm.csv')


train_val_emceb = pd.read_csv(data_path_emceb_train_val)
test_emceb = pd.read_csv(data_path_emceb_test)
train_val_emceb = train_val_emceb.rename(columns={ train_val_emceb.columns[4]: 'EMCEB CN relative probability', train_val_emceb.columns[5]: 'EMCEB MCI relative probability', train_val_emceb.columns[6]: 'EMCEB AD relative probability' })
test_emceb = test_emceb.rename(columns={test_emceb.columns[5]: 'EMCEB CN relative probability', test_emceb.columns[6]: 'EMCEB MCI relative probability', test_emceb.columns[7]: 'EMCEB AD relative probability' })
train_val_emceb = train_val_emceb.sort_values(by=['RID', 'Forecast Date'])
test_emceb = test_emceb.sort_values(by=['RID', 'Forecast Date'])


train_val_blv = pd.read_csv(data_path_blv_train_val)
test_blv = pd.read_csv(data_path_blv_test)
train_val_blv = train_val_blv.rename(columns={train_val_blv.columns[4]: 'BLV CN relative probability', train_val_blv.columns[5]: 'BLV MCI relative probability', train_val_blv.columns[6]: 'BLV AD relative probability' })
test_blv = test_blv.rename(columns={test_blv.columns[5]: 'BLV CN relative probability', test_blv.columns[6]: 'BLV MCI relative probability', test_blv.columns[7]: 'BLV AD relative probability' })
train_val_blv = train_val_blv.sort_values(by=['RID', 'Forecast Date'])
test_blv = test_blv.sort_values(by=['RID', 'Forecast Date'])


train_val_bsvm = pd.read_csv(data_path_bsvm_train_val)
test_bsvm = pd.read_csv(data_path_bsvm_test)
train_val_bsvm = train_val_bsvm.rename(columns={train_val_bsvm.columns[4]: 'BSVM CN relative probability', train_val_bsvm.columns[5]: 'BSVM MCI relative probability', train_val_bsvm.columns[6]: 'BSVM AD relative probability' })
test_bsvm = test_bsvm.rename(columns={test_bsvm.columns[5]: 'BSVM CN relative probability', test_bsvm.columns[6]: 'BSVM MCI relative probability', test_bsvm.columns[7]: 'BSVM AD relative probability' })
train_val_bsvm = train_val_bsvm.sort_values(by=['RID', 'Forecast Date'])
test_bsvm = test_bsvm.sort_values(by=['RID', 'Forecast Date'])


predictions_train_val = pd.concat([train_val_blv, train_val_bsvm, train_val_emceb], axis=1)
predictions_test = pd.concat([test_blv, test_bsvm, test_emceb], axis=1)

# Obtain true labels
true_labels = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/benchmark_last_visit/Random_split_train_val/truelabels.csv')
true_labels = true_labels.drop(columns={'Unnamed: 0'})
CONVERTERS = true_labels[['CONVERTERS']]
true_labels = true_labels.drop(columns='CONVERTERS')
true_labels = true_labels.sort_values(by=['RID', 'EXAMDATE'])
# Save which ones are converters (0 = stable, 1 = converter)
converters = true_labels.copy()
converters = converters.replace({'DXCHANGE': {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}})
converters = converters.rename(columns={"DXCHANGE": "Converters"})
converters = converters['Converters']
converters = converters.fillna(0)


# Preprocess true labels
true_labels = true_labels.replace({'DXCHANGE': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})
true_labels = true_labels.rename(columns={"DXCHANGE": "Diagnosis"})


h = list(true_labels)
remove_columns = h[1:8] + [h[9]] + h[14:17] + h[45:47] + h[53:73] + h[74:486] + h[832:838] + h[1172:1174] + \
h[1657:1667] + h[1895:1902] + h[1905:]
true_labels: pd.DataFrame = true_labels.drop(remove_columns, axis=1)

h = list(true_labels)
for i in range(5, len(h)):
    if true_labels[h[i]].dtype != 'float64':
        true_labels[h[i]] = pd.to_numeric(true_labels[h[i]], errors='coerce')

# Ventricles_ICV = Ventricles/ICV_bl. So make sure ICV_bl is not zero to avoid division by zero
icv_bl_median = true_labels['ICV_bl'].median()
true_labels.loc[true_labels['ICV_bl'] == 0, 'ICV_bl'] = icv_bl_median

if 'Ventricles_ICV' not in true_labels.columns:
    true_labels["Ventricles_ICV"] = true_labels["Ventricles"].values / true_labels["ICV_bl"].values

if 'APOE4' in true_labels.columns: 
    true_labels = true_labels.drop(['EXAMDATE', 'PTGENDER', 'PTEDUCAT', 'APOE4'], axis=1)
else: 
    true_labels = true_labels.drop(['EXAMDATE', 'PTGENDER', 'PTEDUCAT'], axis=1)

true_labels = EMCEB.fill_nans_by_older_values(true_labels)
true_labels = true_labels[['Diagnosis', 'ADAS13', "Ventricles_ICV"]]

# Obtain test predictions
dataPathD4 = "jupyter/data/TADPOLE_D4_corr.csv"

# Obtain true labels
true_labels_test, converters_test = get_true_labels(dataPathD4)
true_labels_test =true_labels_test.sort_values(by=['RID'])
########################################################################
diagnosis_columns = ['BLV CN relative probability', 'BLV MCI relative probability',
                'BLV AD relative probability', 'BSVM CN relative probability',
                'BSVM MCI relative probability', 'BSVM AD relative probability', 'EMCEB CN relative probability',
                'EMCEB MCI relative probability', 'EMCEB AD relative probability']


x_test = predictions_test[diagnosis_columns]
y_test = true_labels_test['Diagnosis']
x_train_val = predictions_train_val[diagnosis_columns]
y_train_val =true_labels[['Diagnosis']]

x_test = pd.concat([x_test, converters_test], axis=1)
xy_test = pd.concat([x_test, y_test], axis=1)
xy_test = xy_test.dropna()

## for exp5 
X_IND_TEST['Diagnosis2'] = X_IND_TEST["Diagnosis"]
X_IND_TEST = X_IND_TEST.drop(columns= 'Diagnosis')
X_TEST['Diagnosis2'] = X_TEST["Diagnosis"]
X_TEST = X_TEST.drop(columns= 'Diagnosis')
xy_test_exp5 = pd.concat([x_test, y_test, X_TEST], axis=1)
xy_test_exp5 = xy_test_exp5[xy_test_exp5['Diagnosis2'].notna()]

xy_test = xy_test.dropna()

# Drop the same columns for exp 5 and 7
xy_test['index2'] = xy_test.index
xy_test_exp5['index2'] = xy_test_exp5.index
output = pd.DataFrame()
for s in range(len(xy_test)):
        currSubjMask = xy_test['index2'].iloc[s] == xy_test_exp5['index2']
        currSubjData = xy_test_exp5[currSubjMask]
        output = output.append(currSubjData)
xy_test_exp5=output
xy_test = xy_test.drop(columns='index2')

xy_test_exp5 = xy_test_exp5.drop(columns='index2')


xy_train_val = pd.concat([x_train_val, y_train_val, X_IND_TEST], axis=1)
xy_train_val = xy_train_val[xy_train_val['Diagnosis'].notna()]
#xy_train_val = xy_train_val.dropna()
# Drop the same indexes that were dropped in true_labels and predictions_train_val
msk = xy_train_val.index
converters = converters[msk]
# msk = xy_test.index
# converters_test = converters_test[msk]

x_test = xy_test.drop(columns= ['Diagnosis'])
y_test = xy_test[['Diagnosis']]

converters_test = xy_test[["Converters"]]
x_train_val = xy_train_val[['BLV CN relative probability', 'BLV MCI relative probability',
                'BLV AD relative probability', 'BSVM CN relative probability',
                'BSVM MCI relative probability', 'BSVM AD relative probability', 'EMCEB CN relative probability',
                'EMCEB MCI relative probability', 'EMCEB AD relative probability']]
y_train_val = xy_train_val[['Diagnosis']]
y_train_val = y_train_val.replace({'Diagnosis': {1: 0, 2: 1, 3: 2}})

# Split random and stratified on Diagnosis and Converters in 90% and 10% respectively
df = pd.concat([x_train_val, y_train_val, converters], axis=1)
xy_converters_train, xy_converters_val = train_test_split(df, test_size=0.1, random_state=0, stratify=df[['Diagnosis', 'Converters']])

#x_train, x_val, y_train, y_val = split_train_val_NN(x_train_val, y_train_val, converters, kfold=1, val_size=0.1)

# Make if statment where can be decided if converters should be input for NN

x_train = xy_converters_train.drop(columns= ['Diagnosis'])
y_train = xy_converters_train[['Diagnosis']]
#converters_train = xy_converters_train[['Converters']]
x_val = xy_converters_val.drop(columns= ['Diagnosis'])
y_val = xy_converters_val[['Diagnosis']]
#converters_val = xy_converters_val[['Converters']]



x_train.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_train.csv')
y_train.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_train.csv')
x_val.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_val.csv')
y_val.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_val.csv')
x_test.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_test.csv')
y_test.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_test.csv')
#converters.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/converters_train_val.csv')
#converters_train.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/converters_train.csv')
#converters_val.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/converters_val.csv')
#converters_test.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/converters_test.csv')



############### FOR EXP 5
# Split random and stratified on Diagnosis and Converters in 90% and 10% respectively
##for exp5
x_test= xy_test_exp5.drop(columns= ['Diagnosis', 'Diagnosis2'])
y_test = xy_test_exp5[['Diagnosis']]



df = pd.concat([x_train_val, y_train_val, converters, X_IND_TEST], axis=1)
# fill left over nans with 0
df = df.fillna(-0.05)
x_test = x_test.fillna(-0.05)

xy_converters_train, xy_converters_val = train_test_split(df, test_size=0.1, random_state=0, stratify=df[['Diagnosis', 'Converters']])

#x_train, x_val, y_train, y_val = split_train_val_NN(x_train_val, y_train_val, converters, kfold=1, val_size=0.1)

# Make if statment where can be decided if converters should be input for NN

x_train = xy_converters_train.drop(columns= ['Diagnosis','Diagnosis2'])
y_train = xy_converters_train[['Diagnosis']]
#converters_train = xy_converters_train[['Converters']]
x_val = xy_converters_val.drop(columns= ['Diagnosis','Diagnosis2'])
y_val = xy_converters_val[['Diagnosis']]
#converters_val = xy_converters_val[['Converters']]


# FOR Exp 8: All data
# concat x and y and split in train and val
Y_TRAIN = Y_TRAIN['Diagnosis']
X_TRAIN = X_TRAIN.drop(columns = 'Diagnosis')
XY_train_val = pd.concat([X_TRAIN, Y_TRAIN], axis=1)
XY_train_val = XY_train_val[XY_train_val['Diagnosis'].notna()]
XY_train, XY_val = train_test_split(XY_train_val, test_size=0.1, random_state=0, stratify=XY_train_val[['Diagnosis']])

X_train = XY_train.drop(columns= 'Diagnosis')
Y_train = XY_train[['Diagnosis']]
X_val = XY_val.drop(columns= 'Diagnosis')
Y_val = XY_val[['Diagnosis']]
Y_train = Y_train.replace({'Diagnosis': {1: 0, 2: 1, 3: 2}})
Y_val = Y_val.replace({'Diagnosis': {1: 0, 2: 1, 3: 2}})

X_train = X_train.fillna(-0.05)
X_val = X_val.fillna(-0.05)



X_train.to_csv('tadpole_algorithms/tests/Outputs/ResNet/X_TRAIN.csv')
Y_train.to_csv('tadpole_algorithms/tests/Outputs/ResNet/Y_TRAIN.csv')
X_val.to_csv('tadpole_algorithms/tests/Outputs/ResNet/X_VAL.csv')
Y_val.to_csv('tadpole_algorithms/tests/Outputs/ResNet/Y_VAL.csv')

x_train.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_train_exp51.csv')
y_train.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_train_exp51.csv')
x_val.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_val_exp51.csv')
y_val.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_val_exp51.csv')
x_test.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_test_exp51.csv')
y_test.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_test_exp51.csv')

predictions_test = x_train.drop(columns={'index', 'Converters','BLV CN relative probability', 'BLV MCI relative probability', 'BLV AD relative probability', 'BSVM CN relative probability', 
                                "BSVM MCI relative probability", 'BSVM AD relative probability', 'EMCEB CN relative probability','EMCEB MCI relative probability', 
                                'EMCEB AD relative probability', 'Unnamed: 0'})
predictions_test.to_csv('tadpole_algorithms/tests/Outputs/ResNet/X_TEST.csv')