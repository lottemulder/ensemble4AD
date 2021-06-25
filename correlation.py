import pandas as pd
# from sklearn import 

# Individual models single traininset scenario
blv = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/blv_100p.csv')
bsvm = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/bsvm100p.csv')
emceb = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/emceb_100p.csv')
emc1 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/emc1_100p.csv')
btmty = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/btmty_100p.csv')
resnet =pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp8.csv')

#indiviudal models separate trainings set senario
blv_50 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/blv_50p.csv')
bsvm_50 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/bsvm_50p.csv')
emceb_50 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/emceb_50p.csv')
resnet_50 =pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/new_exp5_true_false.csv')

# Learned ensembles separate training set scenario
OneNeuron = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp3.csv')
TwoLayers = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp4.csv')
DoubleResNet =pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp6.csv')
ResNetFuser1 =pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp7.csv')
ResNetFuser2 =pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/exp9.csv')

# Un-learned ensembles single training set scenario
MeanBest = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/df_mean_resnet_emc1_sc1.csv')
MeanAll = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/df_mean_resnet_btmty_emc1_emceb_bsvm_blv_sc1.csv')
MedianBest = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/median_emc1_resnet_sc1.csv')
MedianAll = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/median_emceb_blv_bsvm_resnet_emc1_btmty_sc1.csv')

# Un-learned ensembles separate training set scenario
MeanBest_50 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/df_mean_resnet_emceb_sc2.csv')
MeanAll_50 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/df_mean_resnet_emceb_bsvm_blv_sc2.csv')
MedianBest_50 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/median_emceb_resnet_sc2.csv')
MedianAll_50 = pd.read_csv('tadpole_algorithms/tests/Outputs/CombinationData/NN_poging2_alles_random_5050split_train_test/significance_testing/median_emceb_blv_bsvm_resnet_sc2.csv')

# Individual models single traininset scenario
blv = blv.drop(columns= 'Unnamed: 0')
bsvm = bsvm.drop(columns= 'Unnamed: 0')
emceb = emceb.drop(columns= 'Unnamed: 0')
emc1 = emc1.drop(columns= 'Unnamed: 0')
btmty = btmty.drop(columns= 'Unnamed: 0')
resnet = resnet.drop(columns= 'Unnamed: 0')

#indiviudal models separate trainings set senario
blv_50 = blv_50.drop(columns= 'Unnamed: 0')
bsvm_50 = bsvm_50.drop(columns= 'Unnamed: 0')
emceb_50 = emceb_50.drop(columns= 'Unnamed: 0')
resnet_50 = resnet_50.drop(columns= 'Unnamed: 0')

# Learned ensembles separate training set scenario
OneNeuron = OneNeuron.drop(columns= 'Unnamed: 0')
TwoLayers = TwoLayers.drop(columns= 'Unnamed: 0')
DoubleResNet = DoubleResNet.drop(columns= 'Unnamed: 0')
ResNetFuser1 = ResNetFuser1.drop(columns= 'Unnamed: 0')
ResNetFuser2 = ResNetFuser2.drop(columns= 'Unnamed: 0')

# Un-learned ensembles single training set scenario
MeanBest = MeanBest.drop(columns= 'Unnamed: 0')
MeanAll = MeanAll.drop(columns= 'Unnamed: 0')
MedianBest = MedianBest.drop(columns= 'Unnamed: 0')
MedianAll = MedianAll.drop(columns= 'Unnamed: 0')

# Un-learned ensembles separate training set scenario
MeanBest_50 = MedianBest_50.drop(columns= 'Unnamed: 0')
MeanAll_50 = MedianAll_50.drop(columns= 'Unnamed: 0')
MedianBest_50 = MedianBest_50.drop(columns= 'Unnamed: 0')
MedianAll_50 = MedianAll_50.drop(columns= 'Unnamed: 0')

print(emceb_50.sum())

df = pd.concat((blv_50, bsvm_50, emceb_50, resnet_50, MeanAll_50, MeanBest_50, MedianAll_50, MedianBest_50, OneNeuron, TwoLayers, ResNetFuser1, ResNetFuser2, DoubleResNet), axis=1)
df.columns = ['BLV', 'BSVM', 'EMCEB', 'ResNet', 'MeanAll', 'MeanBest', 'MedianAll', 'MedianBest', 'OneNeuron', 'TwoLayers', 'ResNetFuser1', 'ResNetFuser2', 'DoubleResNet']


print(df.corr(method='spearman').round(3))

df1 = pd.concat((blv, bsvm, emceb, emc1, btmty, resnet_50), axis=1)
df1.columns = ['BLV', 'BSVM', 'EMCEB', 'EMC1', 'BTMTY', 'ResNet']
print(df1.corr().round(3))