# GIST - AI Project - Celso (reference: http://www.science.smith.edu/dftwiki/index.php/Tutorial:_Playing_with_the_Boston_Housing_Data)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

# Import data
COLUMNS = ["pol", "s6h", "s3", "s5", "s4h", "ydb"]
FEATURES = ["pol", "s6h", "s3", "s5", "s4h"]
LABEL = "ydb"

training_set = pd.read_csv("metamaterialxy_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("metamaterialxy_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("metamaterialxy_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
ydb_pred = prediction_set.ydb.tolist()

# Define feature columns
feature_cols = [tf.contrib.layers.real_valued_column(k)
                for k in FEATURES]

# Define regressor
regressor = tf.contrib.learn.DNNRegressor(
    feature_columns=feature_cols,
    hidden_units=[600, 400],
    optimizer=tf.train.AdamOptimizer(
                                    learning_rate=0.001,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-08,
                                    ),
    activation_fn=tf.nn.swish,
    dropout=0.3,
    model_dir='/tmp/proj_final')

# Define input_fn
def input_fn(data_set):
    feature_cols = {
        k: tf.constant(
            data_set[k].values,
            # NOTE: define shape explicitly to get rid of the warning
            # Input to shape should be a list
            shape=[data_set[k].size, 1]
        ) for k in FEATURES
    }
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels

# Train the model
regressor.fit(input_fn=lambda: input_fn(training_set), steps=50000)

# Evaluate the model
ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)

loss_score = ev['loss']
print('Loss: {0:f}'.format(loss_score))

# Make Predictions
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
predictions = list(itertools.islice(y, 194))
print('Predictions: {}'.format(str(predictions)))

MSE = metrics.mean_squared_error(ydb_pred, predictions)
print(MSE)

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

# solution
a, b = best_fit(ydb_pred, predictions)
#best fit line:
#y = 0.80 + 0.92x

print(ydb_pred)
print(predictions)

# plot points and fit line
plt.scatter(ydb_pred, predictions)
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Real vs Predicted Values for Resonance Peak")
yfit = [a + b * xi for xi in ydb_pred]
plt.plot(ydb_pred, yfit)

plt.savefig('test-1.png')
plt.show()
