import numpy as np
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

from keras.models import Sequential
from keras.layers import Dense, Dropout
import gcb535challengedata as gcb535
gcb535_data = gcb535.read_data_sets(1, validation_fraction=0.)

model = Sequential()
model.add(Dense(300, input_dim=200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
		optimizer='adagrad',
		metrics=['accuracy'])

model.fit(gcb535_data.train.features,
	gcb535_data.train.labels,
	nb_epoch=100,
	batch_size=50)

res = model.evaluate(gcb535_data.test.features, gcb535_data.test.labels,batch_size=50)
print 'Test Loss: %s, Accuracy: %s' %(res[0], res[1])




