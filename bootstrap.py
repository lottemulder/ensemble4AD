import pandas as pd
import numpy as np

# inititalize dfs for bca and mauc for different bootstraps
results = pd.DataFrame()
bca_df = np.zeros(shape =(100))
mauc_df = np.zeros(shape =(100))

# Concatenate preidcitons and true labels so that resampling happens to same way for both dfs
total_predictions_boot = total_predictions_boot.rename(columns = {0: 'boot_x'})
boot = pd.concat([total_predictions_boot, total_y], axis=1)

# bootstrap 100x on test set, calculate mauc and bca and save in bca_df and mauc_df
for i in range(0,100):
    boot_res = resample(boot, replace=True)
    boot_x = boot_res.drop(columns= 0)
    boot_y = boot_res.drop(columns= 'boot_x')
    boot_y = boot_y.squeeze()
    boot_x = boot_x.to_numpy().squeeze()

    zipTrueLabelAndProbs_res = resample(zipTrueLabelAndProbs)
    mAUC = MAUC.MAUC(zipTrueLabelAndProbs_res, num_classes=num_classes)

    bca = calcBCA(boot_x, boot_y, num_classes)

    print(f'BCA is {bca:.4f} and mAUC is {mAUC:.4f}')
    #print(balnc)

    bca_df[i] = bca
    mauc_df[i] = mAUC

mauc_df = pd.DataFrame(mauc_df, columns = ['mAUC'])
bca_df = pd.DataFrame(bca_df, columns = ['BCA'])
results = pd.concat([bca_df, mauc_df], axis=1)

return bca, mAUC, results
