# ensemble4AD

##### All scripts necassary for making ensembles

tadpole_algorithms.models.BLV.py 			imports model BLV (similar function for other models)	

tadpole_algorithms.preprocessing.split.py 		contains split functions for data splits (used in ind_algs.py)

eval.py							calculates BCA and MAUC (compatible with different test datasets)

tadpole_algorithms.evaluation.MAUC			function 

predict_for_NN_benchmark_lv.py 				predict for the individual algorithms to obtain the training and validation data 

get_test_predictions.py 				predict on X_test to obtain raw X_NN_test

ind_algs.py 						functions used in get_test_predictions.py

concat_predictions.py 					concatinates the predictions of individual algorotihms. Then obtains predictions_train predictions_val and predictions_test 
							which are X_NN_train, X_NN_val and X_NN_test respectively

neural_net.pre_processing.py				function used in get_test_predictions.py to obtain predictions for individual algorithms and in concat_predictions.py to 
							obtain true labels		
