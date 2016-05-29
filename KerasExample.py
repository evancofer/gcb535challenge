import numpy as np
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l1l2, activity_l1l2
import gcb535challengedata as gcb535
gcb535_data = gcb535.read_data_sets(1)

model = Sequential()
model.add(Dense(300, input_dim=200, activation='relu',W_regularizer=l1l2(l1=0., l2=0.) ))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
		optimizer='rmsprop',
		metrics=['accuracy'])

model.fit(gcb535_data.train.features,
	gcb535_data.train.labels,
	nb_epoch=100,
	batch_size=50)

model.evaluate(gcb535_data.test.features, gcb535_data.test.labels,batch_size=50)
###import dataset:




