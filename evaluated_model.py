from neural_net_comb import ADNIdataset, NN, exp_lr_scheduler, check_accuracy, reduce_lr_on_plateau
from tadpole_algorithms.tests.neuralnet.eval import evaluate_forecastt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from tadpole_algorithms.evaluation import calcBCA
from tadpole_algorithms.evaluation import MAUC
from sklearn.utils import resample

test_data = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_test_exp5.csv')
# test_data = test_data.drop(columns = {'Unnamed: 0', 'BLV CN relative probability',
#        'BLV MCI relative probability', 'BLV AD relative probability',
#        'BSVM CN relative probability', 'BSVM MCI relative probability',
#        'BSVM AD relative probability', 'EMCEB CN relative probability',
#        'EMCEB MCI relative probability', 'EMCEB AD relative probability', 'Converters', 'index'})
test_data = test_data.drop(columns = {'Unnamed: 0','Converters', 'index'})



test_data = test_data.to_numpy()
# print(test_data)
test_data = torch.from_numpy(test_data).float()

# input_size = 212
num_classes = 3

input_size= 203
n_layers= 10
n_nodes= 50
dropout= 0.05
nn_type= 4
stop_on_lr_le= 1e-05
batch_size= 10
epochs_not_best_limit= 5


# model = NN(input_size=input_size, num_classes=num_classes, n_layers=n_layers, n_nodes=n_nodes, dropout=dropout, nn_type=nn_type)
# model.load_state_dict(torch.load('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/very_best-model.pt'))
model = torch.load('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/best-model.pt')
model.eval()

with torch.no_grad():
    output = model(test_data)
    m = nn.Softmax(dim=1)
    output = m(output)

output= output.numpy()
output = pd.DataFrame(output, columns= ['CN relative probability', 'MCI relative probability', 'AD relative probability'])
print(output)
 
# Load evaluation dataframe (ADNI-D4)
data_path_eval = ("jupyter/data/TADPOLE_D4_corr.csv")
data_df_eval = pd.read_csv(data_path_eval)
eval_df = data_df_eval

# output.to_csv('tadpole_algorithms/tests/Outputs/ResNet/total_scores_ordered_exp51.csv')
_, _, results, TrueFalse = evaluate_forecastt(eval_df, output)
print(TrueFalse)
TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/new_exp5_true_false.csv')
