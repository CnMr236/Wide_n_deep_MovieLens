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

#1M-MovieLens Data Input
dataframe = pd.read_csv("/Data/userdata.csv", names=COLUMNS, skipinitialspace=True, skiprows=1, engine='python')

rows = len(dataframe)
dataframe = dataframe.iloc[np.random.permutation(rows)].reset_index(drop=True)
dataframe_train = dataframe 

#Datainput: Daten werden in tensors umgewandelt 
def input_fn(data_set):
	feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
	feature_cols = dict(feature_cols)
	labels = tf.constant(data_set[LABEL].values)
	return feature_cols, labels
	
def main(unused_argv):

	df_train = dataframe_train
   
    #Defining Features
	userid = tf.contrib.layers.sparse_column_with_integerized_feature("userid", bucket_size=1000)
	movieid = tf.contrib.layers.sparse_column_with_integerized_feature("movieid", bucket_size=1000)  

	feature_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in FEATURES]
    
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
    
    #Model fitting - Training 
	reg.fit(input_fn=lambda: input_fn(df_train), steps=2000)
    
    #Model fitting - Evaluation
	ev = reg.evaluate(input_fn=lambda: input_fn(df_train), steps=1)
	loss_score = ev["loss"]
	print("Loss Evaluation: {0:f}".format(loss_score))
	print("Training completed")
    
if __name__ == "__main__":
    with tf.device("/cpu:0"): #oder "/gpu:0"
        tf.app.run()
    
    
