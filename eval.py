import numpy as np
import pandas as pd
from datetime import datetime
from tadpole_algorithms.evaluation import MAUC
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample



def calcBCA(estimLabels, trueLabels, nrClasses):
    # Balanced Classification Accuracy

  bcaAll = []
  for c0 in range(nrClasses):
    # c0 can be either CTL, MCI or AD

    # one example when c0=CTL
    # TP - label was estimated as CTL, and the true label was also CTL
    # FP - label was estimated as CTL, but the true label was not CTL (was either MCI or AD).
    TP = np.sum((estimLabels == c0) & (trueLabels == c0))
    TN = np.sum((estimLabels != c0) & (trueLabels != c0))
    FP = np.sum((estimLabels == c0) & (trueLabels != c0))
    FN = np.sum((estimLabels != c0) & (trueLabels == c0))

    # sometimes the sensitivity of specificity can be NaN, if the user doesn't forecast one of the classes.
    # In this case we assume a default value for sensitivity/specificity
    if (TP+FN) == 0:
      sensitivity = 0.5
    else:
      sensitivity = (1. * TP)/(TP+FN)

    if (TN+FP) == 0:
      specificity = 0.5
    else:
      specificity = (1. * TN)/(TN+FP)

    bcaCurr = 0.5*(sensitivity+specificity)
    bcaAll += [bcaCurr]
    # print('bcaCurr %f TP %f TN %f FP %f FN %f' % (bcaCurr, TP, TN, FP, FN))

  return np.mean(bcaAll)


def parseData(d4Df, forecastDf):
  d4Df = d4Df.dropna(subset = {'Diagnosis'})
  trueDiag = d4Df['Diagnosis']
  trueDiag = trueDiag.dropna()
  # forecastDf = forecastDf.reset_index(drop=True)
  # forecastDf['index'] = forecastDf.index
  # output = pd.DataFrame()
  # d4Df['index'] = d4Df.index
  # for s in range(len(d4Df)):
  #     currSubjMask = d4Df['index'].iloc[s] == forecastDf['index']
  #     currSubjData = forecastDf[currSubjMask]
  #     output = output.append(currSubjData)
  # forecastDf=output
  # forecastDf = forecastDf.drop(columns = 'index')
  nrSubj = len(trueDiag)

  zipTrueLabelAndProbs = []

  hardEstimClass = -1 * np.ones(nrSubj, int)

  # for each subject in D4 match the closest user forecasts
  for s in range(nrSubj):
    pCN = forecastDf['CN relative probability'].iloc[s]
    pMCI = forecastDf['MCI relative probability'].iloc[s]
    pAD = forecastDf['AD relative probability'].iloc[s]

    # normalise the relative probabilities by their sum
    pSum = (pCN + pMCI + pAD) / 3
    pCN /= pSum
    pMCI /= pSum
    pAD /= pSum

    hardEstimClass[s] = np.argmax([pCN, pMCI, pAD])

    if isinstance(trueDiag.iloc[s], str) or not np.isnan(trueDiag.iloc[s]):
      zipTrueLabelAndProbs += [(trueDiag.iloc[s], [pCN, pMCI, pAD])]



  # assert trueDiag.shape[0] == hardEstimClass.shape[0]
    
  return zipTrueLabelAndProbs, hardEstimClass, trueDiag


def evaluate_forecastt(d4Df, forecastDf):
  if 'Diagnosis' not in d4Df.columns:
      d4Df = d4Df.replace({'DXCHANGE': {
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
      d4Df = d4Df.replace({'DXCHANGE': {
          'CN': 0,
          'MCI': 1,
          'AD': 2,
      }})
      d4Df = d4Df.rename(columns={"DXCHANGE": "Diagnosis"})
  else:
      d4Df = d4Df.replace({'Diagnosis': {
          'CN': 0,
          'MCI': 1,
          'AD': 2,
      }})       
  

  results = pd.DataFrame()
  bca_df = np.zeros(shape =(100))
  mauc_df = np.zeros(shape =(100))

  zipTrueLabelAndProbs, hardEstimClass, trueDiag = parseData(d4Df, forecastDf)
        
  zipTrueLabelAndProbs = list(zipTrueLabelAndProbs)
  hardEstimClass = pd.DataFrame(hardEstimClass)
  trueDiag = pd.DataFrame(trueDiag)
  trueDiag = trueDiag.reset_index(drop=True)


  boot = pd.concat([hardEstimClass, trueDiag], axis=1)
  boot = boot.dropna()

  #### For mcnemar statistical testing
  TrueFalse = (boot['Diagnosis'] == boot[0])*1
  ##### Multiclass AUC (mAUC) #####
  nrClasses = 3
    
  for i in range(0,100):

    boot_res = resample(boot, replace=True)
    hardEstimClass = boot_res.drop(columns= 'Diagnosis')
    trueDiag = boot_res.drop(columns= 0)
    trueDiag = trueDiag.to_numpy().squeeze()
    hardEstimClass = hardEstimClass.squeeze()


    # Resample true labels and probabilities
    zipTrueLabelAndProbs_res = resample(zipTrueLabelAndProbs)

    mAUC = MAUC.MAUC(zipTrueLabelAndProbs_res, num_classes=nrClasses)


    # print('trueDiagFilt', np.unique(trueDiagFilt), trueDiagFilt)
    bca = calcBCA(hardEstimClass, trueDiag, nrClasses=nrClasses)

    bca_df[i] = bca
    mauc_df[i] = mAUC

  mauc_df = pd.DataFrame(mauc_df, columns = ['mAUC'])
  bca_df = pd.DataFrame(bca_df, columns = ['BCA'])
  results = pd.concat([bca_df, mauc_df], axis=1)

  return bca, mAUC, results, TrueFalse