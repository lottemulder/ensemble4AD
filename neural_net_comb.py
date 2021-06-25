# Imports
import torch as torch
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


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes, n_layers, n_nodes, dropout, nn_type):
        super(NN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.n_layers = n_layers
        self.n_nodes = n_nodes

        self.dropout = dropout
        self.nn_type = nn_type

        if self.nn_type == 0:
            self.initialize_network_resnet_1()
        elif self.nn_type == 1:
            self.initialize_network_resnet_2()
        elif self.nn_type == 2:
            self.initialize_network_resnet_eigen()
        elif self.nn_type == 3:
            self.initialize_network_resnet_fuser()
        elif self.nn_type == 4:
            self.initialize_network_resnet_eigen_all_traindata()
        elif self.nn_type == 5:
            self.initialize_network_one_neuron()
        elif self.nn_type == 6:
            self.initialize_network_few_neurons()

        

    
        

    def forward(self, x):
        if self.nn_type == 0:
            x = self.foward_network_resnet_1(x)
        elif self.nn_type == 1:
            x = self.foward_network_resnet_2(x)
        elif self.nn_type == 2:
            x = self.foward_network_resnet_eigen(x)
        elif self.nn_type == 3:
            x = self.foward_network_resnet_fuser(x)
        elif self.nn_type == 4:
            x = self.foward_network_resnet_eigen_all_traindata(x)
        elif self.nn_type == 5:
            x = self.foward_network_one_neuron(x)
        elif self.nn_type == 6:
            x = self.foward_network_few_neurons(x)


        return x

    def foward_network_one_neuron(self, x):
        x_1 = x[:, :9]

        x = F.relu(self.fc_in(x_1)) + self.skip_in(x_1)
        
        for i in range(self.n_layers):
            x = F.relu(self.bn(self.layer_list[i](x)))
            x = F.relu(self.bn(self.layer_list[i](x)) + x)

            x = self.dropout_layer(x)
            
        x = self.fc_out(x)

        return x

    def foward_network_few_neurons(self, x):
        x_1 = x[:, :9]

        x = F.relu(self.fc_in(x_1)) + self.skip_in(x_1)
        
        for i in range(self.n_layers):
            x = F.relu(self.bn(self.layer_list[i](x)))
            x = F.relu(self.bn(self.layer_list[i](x)) + x)

            x = self.dropout_layer(x)
            
        x = self.fc_out(x)

        return x

    def foward_network_resnet_eigen(self, x):
        x_2 = x[:, 9:]

        x = F.relu(self.fc_in(x_2)) + self.skip_in(x_2)
        
        for i in range(self.n_layers):
            x = F.relu(self.bn(self.layer_list[i](x)))
            x = F.relu(self.bn(self.layer_list[i](x)) + x)

            x = self.dropout_layer(x)
            
        x = self.fc_out(x)

        return x
    
    def foward_network_resnet_eigen_all_traindata(self, x):
        
        x = F.relu(self.fc_in(x)) + self.skip_in(x)
        
        for i in range(self.n_layers):
            x = F.relu(self.bn(self.layer_list[i](x)))
            x = F.relu(self.bn(self.layer_list[i](x)) + x)

            x = self.dropout_layer(x)
            
        x = self.fc_out(x)

        return x



    def foward_network_resnet_1(self, x):

        x = F.relu(self.fc_in(x)) + self.skip_in(x)
        
        for i in range(self.n_layers):
            x = F.relu(self.bn(self.layer_list[i](x)))
            x = F.relu(self.bn(self.layer_list[i](x)) + x)

            x = self.dropout_layer(x)
            
        x = self.fc_out(x)

        return x

    def foward_network_resnet_2(self, x):
        x_1 = x[:, :9]
        x_2 = x[:, 9:]

        x = F.relu(self.fc_in(x_2)) + self.skip_in(x_2)
        
        for i in range(self.n_layers):
            x = F.relu(self.bn(self.layer_list[i](x)))
            x = F.relu(self.bn(self.layer_list[i](x)) + x)

            x = self.dropout_layer(x)
            
        x = self.fc_out(x)

        x = torch.cat((x, x_1), 1)

        x = F.relu(self.fc_in_2(x)) + self.skip_in_2(x)
        
        for i in range(self.n_layers):
            x = F.relu(self.bn(self.layer_list_2[i](x)))
            x = F.relu(self.bn(self.layer_list_2[i](x)) + x)

            x = self.dropout_layer(x)
            
        x = self.fc_out_2(x)

        return x

    def initialize_network_resnet_eigen(self):
        self.fc_in = nn.Linear(self.input_size-9, self.n_nodes)
        self.skip_in = nn.Linear(self.input_size-9, self.n_nodes)
        self.bn = nn.BatchNorm1d(self.n_nodes)
        self.layer_list = list()
        self.layer_2_list = list()
        self.dropout_layer = nn.Dropout(self.dropout)

        for _ in range(self.n_layers):
            self.layer_list.append(nn.Linear(self.n_nodes, self.n_nodes))
            self.layer_2_list.append(nn.Linear(self.n_nodes, self.n_nodes))

        self.fc_out = nn.Linear(self.n_nodes, self.num_classes)
    
    def initialize_network_resnet_eigen_all_traindata(self):
        self.fc_in = nn.Linear(self.input_size, self.n_nodes)
        self.skip_in = nn.Linear(self.input_size, self.n_nodes)
        self.bn = nn.BatchNorm1d(self.n_nodes)
        self.layer_list = list()
        self.layer_2_list = list()
        self.dropout_layer = nn.Dropout(self.dropout)

        for _ in range(self.n_layers):
            self.layer_list.append(nn.Linear(self.n_nodes, self.n_nodes))
            self.layer_2_list.append(nn.Linear(self.n_nodes, self.n_nodes))

        self.fc_out = nn.Linear(self.n_nodes, self.num_classes)


    def initialize_network_resnet_fuser(self):
        self.fc_in = nn.Linear(self.input_size, self.n_nodes)
        self.skip_in = nn.Linear(self.input_size, self.n_nodes)
        self.bn = nn.BatchNorm1d(self.n_nodes)
        self.layer_list = list()
        self.layer_2_list = list()
        self.dropout_layer = nn.Dropout(self.dropout)

        for _ in range(self.n_layers):
            self.layer_list.append(nn.Linear(self.n_nodes, self.n_nodes))
            self.layer_2_list.append(nn.Linear(self.n_nodes, self.n_nodes))

        self.fc_out = nn.Linear(self.n_nodes, self.num_classes)


    def initialize_network_resnet_1(self):
        self.fc_in = nn.Linear(self.input_size, self.n_nodes)
        self.skip_in = nn.Linear(self.input_size, self.n_nodes)
        self.bn = nn.BatchNorm1d(self.n_nodes)
        self.layer_list = list()
        self.layer_2_list = list()
        self.dropout_layer = nn.Dropout(self.dropout)

        for _ in range(self.n_layers):
            self.layer_list.append(nn.Linear(self.n_nodes, self.n_nodes))
            self.layer_2_list.append(nn.Linear(self.n_nodes, self.n_nodes))

        self.fc_out = nn.Linear(self.n_nodes, self.num_classes)

    def initialize_network_resnet_2(self):
        self.fc_in = nn.Linear(self.input_size - 9, self.n_nodes)
        self.skip_in = nn.Linear(self.input_size - 9, self.n_nodes)
        self.bn = nn.BatchNorm1d(self.n_nodes)
        self.layer_list = list()
        self.layer_2_list = list()
        self.dropout_layer = nn.Dropout(self.dropout)

        for _ in range(self.n_layers):
            self.layer_list.append(nn.Linear(self.n_nodes, self.n_nodes))
            self.layer_2_list.append(nn.Linear(self.n_nodes, self.n_nodes))

        self.fc_out = nn.Linear(self.n_nodes, 3)

        self.fc_in_2 = nn.Linear(12, self.n_nodes)
        self.skip_in_2 = nn.Linear(12, self.n_nodes)

        self.layer_list_2 = list()
        self.layer_2_list_2 = list()

        for _ in range(self.n_layers):
            self.layer_list_2.append(nn.Linear(self.n_nodes, self.n_nodes))
            self.layer_2_list_2.append(nn.Linear(self.n_nodes, self.n_nodes))

        self.fc_out_2 = nn.Linear(self.n_nodes, self.num_classes)


#model = NN(15, 3)
#x = torch.randn(5, 15)
#print(model(x).shape)

# Set device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters 
# input_size= 9
# num_classes = 3
# lr = 0.001
# batch_size = 10
# num_epochs = 150

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def reduce_lr_on_plateau(optimizer, val_losses, epochs_not_best_limit):
    
    epochs_not_best = 0
    best_loss = 0
    updated = False

    for loss in val_losses:
        if loss > best_loss:
            best_loss = loss
            epochs_not_best = 0
        else:
            epochs_not_best = epochs_not_best + 1


    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    if epochs_not_best != 0 and epochs_not_best % epochs_not_best_limit == 0:
        lr = lr / 2
        updated = True
    
        print(f'Updated LR to : {lr}')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, updated

# Implement a custom Dataset:
# inherit Dataset
# implement __init__, __getitem__, and __len__

class ADNIdataset(Dataset):
    def __init__(self, path, labelsPath, use_converters):
        # Initialize data, download etc.
        # read with numpy or pandas

        dfx = pd.read_csv(path)
        dfy = pd.read_csv(labelsPath)
        dfx = dfx.drop(columns='Unnamed: 0')
        
        # dfx = dfx.drop(columns='index')

        dfy = dfy.drop(columns='Unnamed: 0')

        if use_converters == False:
            dfx = dfx.drop(columns='Converters')

        self.x = np.array(dfx)
        self.y = np.array(dfy)
        self.n_samples = dfy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# trainPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_train.csv'
# trainLabelsPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_train.csv'
# valPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_val.csv'
# valLabelsPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_val.csv'
# testPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/predictions_test.csv'
# testLabelPath = 'tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/true_labels_test.csv'


# datasetTrain= ADNIdataset(trainPath, trainLabelsPath)
# datasetVal = ADNIdataset(valPath, valLabelsPath)
# datasetTest = ADNIdataset(testPath, testLabelPath)

#first_data = datasetTrain[0]
#features, labels = first_data
#print(features, labels)

# Load data
# train = data_utils.TensorDataset(torch.Tensor(datasetTrain.x), torch.Tensor(datasetTrain.y))
# train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
# val = data_utils.TensorDataset(torch.Tensor(datasetVal.x), torch.Tensor(datasetVal.y))
# val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=True)
# test = data_utils.TensorDataset(torch.Tensor(datasetTest.x), torch.Tensor(datasetTest.y))
# test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)



#total_samples = len(datasetTrain)
#n_iterations = math.ceil(total_samples/batch_size)
#print(total_samples, n_iterations)


# Initialize model
# model = NN(input_size=input_size, num_classes = num_classes)

# # Loss and optimiser
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr = lr)

# data, targets = next(iter(train_loader))

# # Train network
# losses = []

# for epoch in range(num_epochs):
#     running_loss = 0.0
#     optimizer = exp_lr_scheduler(optimizer, epoch)
    
#     for batch_idx, (data, targets) in enumerate(train_loader):

#         #if (batch_idx+1) % 5 ==0:
#         #    print(f'epoch {epoch+1}/{num_epochs}, step{batch_idx+1}/ {n_iterations}, inputs{data.shape}')

#         # Get data to cuda if possible
#         #data = data.to(device=device)
#         #targets = targets.to(device=device)
#         #print(data.shape)
#         # Get to correct shape
#         #data = data.reshape(data.shape[0], -1)
        
#         targets = targets.squeeze(1).long()
        
#         # forward
#         scores = model(data)
#         loss = criterion(scores, targets)
#         #print(loss)

#         #backwared

#         optimizer.zero_grad()
#         loss.backward()

#         # gradient descent or adam step
#         optimizer.step()

#         running_loss += loss.item() * data.size(0)

#     losses.append(running_loss / len(train_loader))
# plt.plot(range(1, num_epochs+1), losses, label= 'Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()



# # Check accuracy, BCA and mAUC on trianing & test set to see how good model is

# # Initialize df for saving metrics

def check_accuracy(loader, model, which_loader, show_plot=False):
    if which_loader == 'train_loader':
        print("Checking accuracy on training data")
    elif which_loader == 'val_loader':
        print("Checking accuracy on validation data")

    else:
        print("Checking accuracy on test data")
    num_correct = 0
    num_samples = 0 
    total_predictions=pd.DataFrame()
    total_y = pd.DataFrame()
    total_scores = pd.DataFrame()
    model.eval()
    # indexes = pd.DataFrame()

    with torch.no_grad():
        for x, y in loader:
            #x = x.to(device=device)
            #y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)
        
            scores = model(x)
            m = nn.Softmax(dim=1)
            scores = m(scores)

            # 64*10
            _, predictions = scores.max(1)
            num_correct += (predictions == y.squeeze(1).long()).sum()
            num_samples += predictions.size(0)
            total_predictions = total_predictions.append(pd.DataFrame(predictions.unsqueeze(dim=1).numpy()))
            total_y = total_y.append(pd.DataFrame(y.squeeze(1).long().numpy()), ignore_index=True)

            # indexes = indexes.append(index)
            total_scores = total_scores.append(pd.DataFrame(scores.numpy()))
            # total_scores = pd.concat(total_scores, indexes, axis=1)
        # Make sure total scores and total labels are in the right format for calculation bca and mAUC with tadpole challange code
        total_scores = total_scores.reset_index()
        total_scores = total_scores.drop(columns='index')
        total_scores = total_scores.rename(columns = {total_scores.columns[0]: 'CN relative probability',
                                                         total_scores.columns[1]:'MCI relative probability',
                                                         total_scores.columns[2]: 'AD relative probability'})

        total_predictions = total_predictions.reset_index()
        total_y = total_y.reset_index()
        total_predictions = total_predictions.drop(columns='index')
        total_y = total_y.drop(columns='index')
        total_y = total_y.squeeze()
        total_predictions_boot = total_predictions.copy()
        total_predictions = total_predictions.to_numpy().squeeze()

        # Calulcate mAUC  
        nrSubj = len(total_predictions)
        hardEstimClass = -1 * np.ones(nrSubj, int)
        zipTrueLabelAndProbs = []

        for s in range(nrSubj):
            pCN = total_scores['CN relative probability'].iloc[s]
            pMCI = total_scores['MCI relative probability'].iloc[s]
            pAD = total_scores['AD relative probability'].iloc[s]

            # normalise the relative probabilities by their sum
            pSum = (pCN + pMCI + pAD) / 3
            pCN /= pSum
            pMCI /= pSum
            pAD /= pSum

            hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])
            zipTrueLabelAndProbs += [(total_y.iloc[s], [pCN, pMCI, pAD])]
        
        zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)

        if which_loader == 'test_loader':

            # inititalize dfs for bca and mauc for different bootstraps
            results = pd.DataFrame()
            bca_df = np.zeros(shape =(100))
            mauc_df = np.zeros(shape =(100))
            
            # Concatenate preidcitons and true labels so that resampling happens to same way for both dfs
            total_predictions_boot = total_predictions_boot.rename(columns = {0: 'boot_x'})
            boot = pd.concat([total_predictions_boot, total_y], axis=1)
            TrueFalse = (boot['boot_x'] == boot[0])*1

            # bootstrap 100x on test set, calculate mauc and bca and save in bca_df and mauc_df
            for i in range(0,100):
                boot_res = resample(boot, replace=True)
                boot_x = boot_res.drop(columns= 0)
                boot_y = boot_res.drop(columns= 'boot_x')
                boot_y = boot_y.squeeze()
                boot_x = boot_x.to_numpy().squeeze()

                zipTrueLabelAndProbs_res = resample(zipTrueLabelAndProbs)
                mAUC = MAUC.MAUC(zipTrueLabelAndProbs_res, num_classes=3)

                bca = calcBCA(boot_x, boot_y, 3)

                print(f'BCA is {bca:.4f} and mAUC is {mAUC:.4f}')
                #print(balnc)

                bca_df[i] = bca
                mauc_df[i] = mAUC
            
            mauc_df = pd.DataFrame(mauc_df, columns = ['mAUC'])
            bca_df = pd.DataFrame(bca_df, columns = ['BCA'])
            results = pd.concat([bca_df, mauc_df], axis=1)

            return bca, mAUC, results, TrueFalse, total_scores

        else:
            bca = calcBCA(total_predictions, total_y, 3)
            print(f'Got {num_correct} / {num_samples} with accuracy  {float(num_correct)/float(num_samples)*100:.2f}')

            mAUC = MAUC.MAUC(zipTrueLabelAndProbs, num_classes=3)
            
            if show_plot:
                # Plot balance in classes in train val and test
                fig = plt.figure()
                ax = fig.add_axes([0,0,1,1])
                lbls = ['CN', 'MCI', 'AD']
                balnc = [sum(total_y==0), sum(total_y==1), sum(total_y==2)]
                ax.bar(x=lbls, height=balnc)

            #plt.show()

            print(f'BCA is {bca:.4f} and mAUC is {mAUC:.4f}')
            # print(balnc)
            return bca, mAUC

    model.train()
    #print(model)

    


# bca_train, mAUC_train = check_accuracy(train_loader, model)
# bca_val, mAUC_val = check_accuracy(val_loader, model)
# _, _, results = check_accuracy(test_loader, model)

# #results = pd.DataFrame([[round(bca_train,4), round(mAUC_train,4)], [round(bca_val,4), round(mAUC_val,4)], [round(bca_test,4), round(mAUC_test,4)]],  columns=['BCA', 'mAUC'], index=['train', 'val', 'test'])

# results.to_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/Results_exp3/results_100_btsr.csv')