from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import io

from six.moves import urllib              

import pandas as pd              
import numpy as np               
import tensorflow as tf          

COLUMNS = ["userid", "movieid", "time", "gender", "age", 
           "occupation", "zip", "title", "genre", "rating"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["gender", "title", "genre", "occupation"]
CONTINUOUS_COLUMNS = ["age","zip","userid", "movieid", "time"]


#1ML-Dataset-Input
ratings = pd.read_csv("//Users/ChrisM/Dataset/ml-1m/ratings.dat", sep="::", names=['userid','movieid','rating','time'], skipinitialspace=True, skiprows=1, engine='python') # Pandas load our dataset as a dataframe
users = pd.read_csv("//Users/ChrisM/Dataset/ml-1m/users.dat", sep="::", names=['userid','gender','age','occupation','zip'], skipinitialspace=True, skiprows=1, engine='python')
movies = pd.read_csv("//Users/ChrisM/Dataset/ml-1m/movies.dat", sep="::", names=['movieid','title','genre'], skipinitialspace=True, skiprows=1, engine='python')
dataframe = pd.merge(pd.merge(ratings,users),movies)
dataframe = dataframe[0:100000] #Only use 100k Rows of Dataset
  
#Splitting Data for train and test           
rows = len(dataframe)
dataframe = dataframe.iloc[np.random.permutation(rows)].reset_index(drop=True)
split_index = int(rows * 0.80)
dataframe_train = dataframe[0:split_index]
dataframe_test = dataframe[split_index:].reset_index(drop=True)
#Drop NaN Elements
dataframe_train = dataframe_train.dropna(how='any', axis=0)
dataframe_test = dataframe_test.dropna(how='any', axis=0)

  
def build_estimator(model_dir, model_type):

    # Sparse base columns.
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["M", "F"])
    title = tf.contrib.layers.sparse_column_with_hash_bucket("title", hash_bucket_size=1000)
    genre = tf.contrib.layers.sparse_column_with_hash_bucket("genre", hash_bucket_size=1000)
    zipcode = tf.contrib.layers.sparse_column_with_hash_bucket("zip", hash_bucket_size=1000)
    occupation = tf.contrib.layers.sparse_column_with_integerized_feature("occupation", bucket_size=1000)
    
    
    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    userid = tf.contrib.layers.real_valued_column("userid")
    movieid = tf.contrib.layers.real_valued_column("movieid")
    time = tf.contrib.layers.real_valued_column("time")
 
  # Transformations.
    #none

  # Wide columns and deep columns.
    wide_columns = [age, userid, movieid, time]
    
    
    deep_columns = [tf.contrib.layers.embedding_column(title, dimension=8),
                    tf.contrib.layers.embedding_column(genre, dimension=8),
                    tf.contrib.layers.embedding_column(gender, dimension=8),
                    tf.contrib.layers.embedding_column(occupation, dimension=8),
                    tf.contrib.layers.embedding_column(zipcode, dimension=8),
                   ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
    return m

def input_fn(dataframe):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(dataframe[k].values, shape=[dataframe[k].size, 1]) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
          k: tf.SparseTensor(
          indices=[[i, 0] for i in range(dataframe[k].size)],
          values=dataframe[k].values,
          dense_shape=[dataframe[k].size, 1])
          for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
    label = tf.constant(dataframe[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
    return feature_cols, label
    
def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    #train_data = dataframe_train
    #test_data = dataframe_test

  # remove NaN elements
    #train_data = dataframe_train.dropna(how='any', axis=0)
    #test_data = dataframe_test.dropna(how='any', axis=0)

    dataframe_train[LABEL_COLUMN] = (
      dataframe_train["rating"].apply(lambda x: "5" == x)).astype(int)
    dataframe_test[LABEL_COLUMN] = (
      dataframe_test["rating"].apply(lambda x: "5" == x)).astype(int)

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(dataframe_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(dataframe_test), steps=1)
    for key in sorted(results):
      print("%s: %s" % (key, results[key]))


FLAGS = None

def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps, FLAGS.train_data, FLAGS.test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
      "--model_dir",
      type=str, 
      default="",
      help="Base directory for output models."
  )
    parser.add_argument(
      "--model_type",
      type=str, 
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
    parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
    parser.add_argument(
      "--train_data",
      type=str, 
      default="",
      help="Path to the training data."
  )
    parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

