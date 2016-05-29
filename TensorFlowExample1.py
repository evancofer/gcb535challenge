import numpy as np
from itertools import repeat
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

###Important variables:
learning_rate = 0.100

###Making the layers:
features = tf.placeholder(tf.float32, shape=(None, 200))
labels = tf.placeholder(tf.float32, shape=(None ,1))


from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1l2, activity_l1l2
import random


model = Sequential()
first_layer = Dense(92, activation='sigmoid', input_shape=(None,200),
			W_regularizer=l1l2(l1=0.034, l2=0.001),
			activity_regularizer=activity_l1l2(l1=0.334, l2=0.001)
			)
first_layer.set_input(features)
model.add(first_layer)
model.add(Dense(1, activation='linear'))
output_layer = model.output

###Objective function:
from keras.objectives import MSE
loss = tf.reduce_mean(MSE(labels, output_layer))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


###import dataset:
import gcb535challengedata as gcb535
gcb535_data = gcb535.read_data_sets(1)

###training:
with sess.as_default():
	for i in range(100):
		batch = gcb535_data.train.next_batch(50)
		train_step.run(feed_dict={
			features: batch[0],
			labels: batch[1],
			K.learning_phase(): 1
		})


###evaluation:
correct_prediction = tf.equal(tf.round(output_layer), labels)
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

with sess.as_default():
		
	print accuracy.eval(feed_dict={
			features: gcb535_data.test.features,
			labels: gcb535_data.test.labels,
			K.learning_phase(): 0
		})

