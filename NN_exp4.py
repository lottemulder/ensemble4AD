from neural_net_comb import ADNIdataset, NN, exp_lr_scheduler, check_accuracy

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



# Paths
trainPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_train.csv'
trainLabelsPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_train.csv'
valPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_val.csv'
valLabelsPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_val.csv'
testPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_test.csv'
testLabelPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_test.csv'

# Paths converters
USE_CONVERTERS = False

# Data sets
datasetTrain= ADNIdataset(trainPath, trainLabelsPath, USE_CONVERTERS)
datasetVal = ADNIdataset(valPath, valLabelsPath, USE_CONVERTERS)
datasetTest = ADNIdataset(testPath, testLabelPath, USE_CONVERTERS)

# Hyperparameters 
input_size= 9
num_classes = 3
lr = 0.0005
batch_size = 5
num_epochs = 150

# Load data
train = data_utils.TensorDataset(torch.Tensor(datasetTrain.x), torch.Tensor(datasetTrain.y))
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
val = data_utils.TensorDataset(torch.Tensor(datasetVal.x), torch.Tensor(datasetVal.y))
val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=True)
test = data_utils.TensorDataset(torch.Tensor(datasetTest.x), torch.Tensor(datasetTest.y))
test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

# Train network
best_model = 0 

for i in range(0,10):

    # Initialize model
    model = NN(input_size=input_size, num_classes = num_classes)

    # Loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    data, targets = next(iter(train_loader))


    losses = []
    

    for epoch in range(num_epochs):
        running_loss = 0.0
        optimizer = exp_lr_scheduler(optimizer, epoch)
        
        for batch_idx, (data, targets) in enumerate(train_loader):

            #if (batch_idx+1) % 5 ==0:
            #    print(f'epoch {epoch+1}/{num_epochs}, step{batch_idx+1}/ {n_iterations}, inputs{data.shape}')

            # Get data to cuda if possible
            #data = data.to(device=device)
            #targets = targets.to(device=device)
            #print(data.shape)
            # Get to correct shape
            #data = data.reshape(data.shape[0], -1)
            
            targets = targets.squeeze(1).long()
            
            # forward
            scores = model(data)
            loss = criterion(scores, targets)
           # print(loss)

            #backwared

            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        losses.append(running_loss / len(train_loader))
        #print('Final loss =', loss)
   
    plt.plot(range(1, num_epochs+1), losses, label= 'Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    bca_train, mAUC_train = check_accuracy(train_loader, model, which_loader = 'train_loader')
    bca_val, mAUC_val = check_accuracy(val_loader, model, which_loader = 'val_loader')
    print('bca', bca_val, 'mauc', mAUC_val)

    model_perf = bca_val + mAUC_val
    if model_perf > best_model:
        torch.save(model.state_dict(), 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp4/best-model.pt') 
        best_model = model_perf

# Check accuracy, BCA and mAUC on trianing & test set to see how good model is
model.load_state_dict(torch.load('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp4/best-model.pt'))
_, _, results, TrueFalse = check_accuracy(test_loader, model, which_loader = 'test_loader')

#results = pd.DataFrame([[round(bca_train,4), round(mAUC_train,4)], [round(bca_val,4), round(mAUC_val,4)], [round(bca_test,4), round(mAUC_test,4)]],  columns=['BCA', 'mAUC'], index=['train', 'val', 'test'])

results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp4/results_100_btsr.csv')
TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp4.csv')

