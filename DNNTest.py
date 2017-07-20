from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

np.random.seed(42)

COLUMNS = ["userid", "movieid", "rating"]
LABEL = "rating"  
FEATURES = ["userid", "movieid"]

#Load Dataset
dataframe_pred = pd.read_csv("Data/testdata.csv", names=COLUMNS, skipinitialspace=True, skiprows=1, engine='python') # Pandas load our dataset as a dataframe

#Datainput: Daten werden in tensors umgewandelt 
def input_fn(data_set):
	feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
	feature_cols = dict(feature_cols)
	labels = tf.constant(data_set[LABEL].values)
	return feature_cols, labels
	
def main(unused_argv): 
	df_pred = dataframe_pred
    
	#Defining Features
	userid = tf.contrib.layers.sparse_column_with_integerized_feature("userid", bucket_size=1000)
	movieid = tf.contrib.layers.sparse_column_with_integerized_feature("movieid", bucket_size=1000)  
    
	featurecat_cols = [tf.contrib.layers.embedding_column(userid, dimension=32),
                       tf.contrib.layers.embedding_column(movieid, dimension=32)
                      ]
    
	#Deep Regressor 3-Hidden Layers
	reg = tf.contrib.learn.DNNRegressor(feature_columns=featurecat_cols,
                                        model_dir="/tmp/MLK00117",
                                        hidden_units=[10, 50, 10],
                                        optimizer=tf.train.ProximalAdagradOptimizer(
                                            learning_rate=0.1,
                                            l1_regularization_strength=0.001)
                                       )  

    
	#Predicting Ratings
	pred = list(reg.predict_scores(input_fn=lambda: input_fn(df_pred)[0], as_iterable=True))
	
    
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		file = open("/Users/ChrisM/TensoRSimulation/Result.txt","w")
		file.write(str(pred))
		file.close()
    	    
if __name__ == "__main__":
	with tf.device("/cpu:0"): #oder "/gpu:0"
		tf.app.run()
