from neural_net_comb import ADNIdataset, NN, exp_lr_scheduler, check_accuracy, reduce_lr_on_plateau

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
init_lr_list =                  [0.005, 0.0005, 0.0005,0.0005, 0.0005]
batch_size_list =               [10, 10,10,10,10]
n_layers_list =                 [10,10,10,10,10]
n_nodes_list =                  [50,50,50,50,50]
dropout_list =                  [0.05, 0.05, 0.05, 0.05, 0.05]
nn_type_list =                  [3,3,3,3,3]
stop_on_lr_le_list =            [1e-5,1e-5,1e-5,1e-5,1e-5]
epochs_not_best_limit_list =    [5,5,5,5,5]

very_best_model = 0
for init_lr, batch_size, n_layers, n_nodes, dropout, nn_type, stop_on_lr_le, epochs_not_best_limit in zip(init_lr_list, batch_size_list, n_layers_list, n_nodes_list, dropout_list, nn_type_list, stop_on_lr_le_list, epochs_not_best_limit_list):

    input_size = 9
    num_classes = 3

    # Train network
    best_model = 0 
    result=pd.DataFrame()
    # Load data
    train = data_utils.TensorDataset(torch.Tensor(datasetTrain.x), torch.Tensor(datasetTrain.y))
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
    val = data_utils.TensorDataset(torch.Tensor(datasetVal.x), torch.Tensor(datasetVal.y))
    val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=True)
    test = data_utils.TensorDataset(torch.Tensor(datasetTest.x), torch.Tensor(datasetTest.y))
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)


    # Initialize model
    model = NN(input_size=input_size, num_classes=num_classes, n_layers=n_layers, n_nodes=n_nodes, dropout=dropout, nn_type=nn_type)

    # Loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = init_lr)

    data, targets = next(iter(train_loader))

    train_losses = []
    train_bca =[]
    train_mAUC=[]
    val_bca = []
    val_mAUC = []
    best_val_bca = 0
    best_val_mAUC = 0

    epochs = list()
    epoch = 0
    if nn_type == 2:
        filename = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/results_nn_type2.txt'
    elif nn_type == 1:
        filename = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/results_nn_type1.txt'
    elif nn_type == 0:
        filename = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/results_nn_type0.txt'
    elif nn_type == 3:
        filename = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/results_nn_type3.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as f:
        f.write(f'hyper parameters\n')

        f.write(f'input_size: {input_size}\n')
        f.write(f'n_layers: {n_layers}\n')
        f.write(f'n_nodes: {n_nodes}\n')
        f.write(f'dropout: {dropout}\n')
        f.write(f'nn_type: {nn_type}\n')
        f.write(f'stop_on_lr_le: {stop_on_lr_le}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'epochs_not_best_limit: {epochs_not_best_limit}\n')

    while True:
        print(f'Epoch: {epoch}...')
        running_loss = 0.0
        optimizer, lr, updated = reduce_lr_on_plateau(optimizer, val_bca, epochs_not_best_limit)
        
        if updated:
            torch.load('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp7/best-model.pt')

        if lr <= stop_on_lr_le:
            print('LR too small. Terminating...')
            break
        
        epochs.append(epoch)

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

        train_losses.append(running_loss / len(train_loader))
        #print('Final loss =', loss)
        bca_train, mAUC_train = check_accuracy(train_loader, model, which_loader = 'train_loader')
        bca_val, mAUC_val = check_accuracy(val_loader, model, which_loader = 'val_loader')

        train_bca.append(bca_train)
        train_mAUC.append(mAUC_train)
        val_bca.append(bca_val)
        val_mAUC.append(mAUC_val)

        if bca_val > best_val_bca:
            best_val_bca = bca_val
            print('Found best so far!')
            torch.save(model, 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp7/best-model.pt') 

        if mAUC_val > best_val_mAUC:
            best_val_mAUC = mAUC_val

        if bca_val > very_best_model:
            very_best_model = bca_val
            print("Found very best model so far!")
            torch.save(model, 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp7/very_best-model.pt') 

        plt.clf()
        plt.plot(epochs, train_bca, label='train')
        plt.plot(epochs, val_bca, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('BCA')
        plt.legend()
        plt.pause(0.0000001)

        epoch = epoch + 1
        
    with open(filename, 'a') as f:
        f.write(f'Results Train\n')
    
        f.write(f'bca per epoch: {train_bca}\n')
        f.write(f'mAUC per epoch: {train_mAUC}\n')


    with open(filename, 'a') as f:
        f.write(f'Results Validation\n')
    
        f.write(f'best BCA: {best_val_bca}\n')
        f.write(f'best mAUC: {best_val_mAUC}\n')
        f.write(f'bca per epoch: {val_bca}\n')
        f.write(f'mAUC per epoch: {val_mAUC}\n')

    # plt.show()
    # print(num_epochs, batch_size, n_nodes, init_lr,'bca', bca_val, 'mauc', mAUC_val)
    # result = result.append([[int(num_epochs),int(batch_size), int(n_nodes), init_lr, bca_train, mAUC_train, bca_val, mAUC_val]])

    # model_perf = bca_val + mAUC_val
    # if model_perf > best_model:
    #     torch.save(model.state_dict(), 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/best-model.pt') 
    #     best_model = model_perf
    # result.columns = [['num_epochs', 'batch_size', 'n_nodes', 'init_lr', 'bca_train', 'mAUC_train', 'bca_val', 'mAUC_val']]
    # result.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/test_results.csv')
    # # Check accuracy, BCA and mAUC on trianing & test set to see how good model is
    torch.load('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp7/best-model.pt')
    model.eval()
    bca_test, mAUC_test, _, _, total_scores = check_accuracy(test_loader, model, which_loader = 'test_loader')

    with open(filename, 'a') as f:
        f.write(f'Results test (do not use to select model)\n')
    
        f.write(f'BCA test: {bca_test}\n')
        f.write(f'mAUC test: {mAUC_test}\n')


        # model_perf = bca_val 
        # if model_perf > very_best_model:
        #     torch.save(model.state_dict(), 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/very_best-model.pt') 
        #     very_best_model = model_perf

# # Check accuracy, BCA and mAUC on trianing & test set to see how good model is
# model.load_state_dict(torch.load('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/very_best-model.pt'))
# _, _, results, TrueFalse = check_accuracy(test_loader, model, which_loader = 'test_loader')

#results = pd.DataFrame([[round(bca_train,4), round(mAUC_train,4)], [round(bca_val,4), round(mAUC_val,4)], [round(bca_test,4), round(mAUC_test,4)]],  columns=['BCA', 'mAUC'], index=['train', 'val', 'test'])
torch.load('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp7/very_best-model.pt')
model.eval()
_, _, results, TrueFalse, scores = check_accuracy(test_loader, model, which_loader = 'test_loader')

results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp7/results_100_btsr.csv')
TrueFalse.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp7.csv')
total_scores.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp7/total_scores.csv')


# results = pd.DataFrame([[round(bca_train,4), round(mAUC_train,4)], [round(bca_val,4), round(mAUC_val,4)], [round(bca_test,4), round(mAUC_test,4)]],  columns=['BCA', 'mAUC'], index=['train', 'val', 'test'])


# plt.plot(range(1, num_epochs+1), losses, label= 'Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# plt.close()


# filename = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Result_exp5/results.txt'

# best_bca = 0
# best_line = 0
# lines = None
# with open(filename, 'r') as f:
#     lines = f.readlines()

#     for i, line in enumerate(lines):
#         if line.startswith('best BCA:'):
#             line_temp = line.replace('best BCA: ', '')

#             current_bca = float(line_temp)

#             if best_bca <= current_bca:
#                 best_bca = current_bca
#                 best_line = i

# nn_type = lines[best_line-8]
# print(best_line, best_bca, lines[best_line-8])