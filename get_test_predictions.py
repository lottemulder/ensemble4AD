from pathlib import Path
import pandas as pd
import numpy as np
from tadpole_algorithms.models import BenchmarkLastVisit
from tadpole_algorithms.models import EMCEB
from tadpole_algorithms.models import BenchmarkSVM
from tadpole_algorithms.models import Benchmark_FRESACAD_R
from tadpole_algorithms.tests.neuralnet.ind_algs import predict_individual_models_and_bootstrap
EMC1= 'no model'

# Load data 
data_path_train_test = Path("jupyter/data/TADPOLE_D1_D2.csv")
data_df_train_test = pd.read_csv(data_path_train_test)

# Load D4 evaluation data set 
data_path_eval = Path("jupyter/data/TADPOLE_D4_corr.csv")
data_df_eval = pd.read_csv(data_path_eval)

predict_individual_models_and_bootstrap(data_df_train_test, data_df_eval, model= BenchmarkLastVisit(), model_name= 'BenchmarkLastVisit', SCENARIO_2=True)
predict_individual_models_and_bootstrap(data_df_train_test, data_df_eval, model= BenchmarkSVM(), model_name= 'BenchmarkSVM', SCENARIO_2=True)
predict_individual_models_and_bootstrap(data_df_train_test, data_df_eval, model= EMCEB(), model_name= 'EMCEB', SCENARIO_2=True)
# predict_individual_models_and_bootstrap(data_df_train_test, data_df_eval, model= EMC1 , model_name= 'EMC1', SCENARIO_2=True)
# predict_individual_models_and_bootstrap(data_df_train_test, data_df_eval, model= Benchmark_FRESACAD_R(), model_name= 'BORREGOSTECMTY', SCENARIO_2=True)
