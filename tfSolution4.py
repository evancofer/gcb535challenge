# All: 0 -> accuracy of 0.495, all 1 -> accuracy of 0.505

def predict(learning_rate_, layer1_wl1, layer1_wl2, layer1_al1, layer1_al2, layer1_nodes, output_activation, input_activation, train_filename, test_filename, verbose, using_keras_accuracy, batch_size_ = 1, train_count_ = 1):
	import numpy as np
	import tensorflow as tf
	from keras import backend as K
	sess = tf.Session()
	K.set_session(sess)
	
#Prepare everything:
	features = tf.placeholder(tf.float32, shape=(None,200))
	labels = tf.placeholder(tf.float32, shape=(None,1))

	#define model and layers:
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.regularizers import l1l2, activity_l1l2
	model = Sequential()
	first_layer = Dense(layer1_nodes, activation=input_activation, input_shape=(None,200),
			W_regularizer=l1l2(l1=layer1_wl1, l2=layer1_wl2), #0.03, 0.001
			activity_regularizer=activity_l1l2(l1=layer1_al1, l2=layer1_al2) #0.03, 0.001
			)
	first_layer.set_input(features)
	model.add(first_layer)

	#add internal layers:
	from keras.layers.noise import GaussianNoise

	model.add(GaussianNoise(0.00000000001))
#	model.add(Dense(25, activation='linear'))	

	#output layer
	model.add(Dense(1, activation=output_activation))
	output_layer = model.output
	

	#load in data:
	train_data = np.loadtxt(open(train_filename, "rb"), delimiter=",")
	train_features = train_data[:, :200]
	train_labels = train_data[:, 200]

	test_data = np.loadtxt(open(test_filename, "rb"), delimiter=",")
	test_features = test_data[:, :200]
	test_labels = test_data[:, 200]

	#batch data:
	batch_count = train_labels.size / batch_size_
	batch_info = "batch size="+str(batch_size_)
	model_info = "learning rate="+str(learning_rate_)+", input nodes="+str(layer1_nodes)
	if not(train_count_ == 1):
		model_info += ", train count="
		model_info += str(train_count_)
	accuracy_info = ""
	regulizer_info = ""
	if not(layer1_al1==0. and layer1_al2==0. and layer1_wl1==0. and layer1_wl2==0.):
		regulizer_info = ", input( wl1="+str(layer1_wl1)+", wl2="+str(layer1_wl2)+", al1="+str(layer1_al1)+", al2="+str(layer1_al2)+")"
 
#Loss function:
	#https://github.com/fchollet/keras/blob/master/keras/objectives.py
	from keras.objectives import mean_squared_error
	loss = tf.reduce_mean(mean_squared_error(labels, output_layer))

#Optimization method:
	train_step = tf.train.GradientDescentOptimizer(learning_rate_).minimize(loss)
	#See https://github.com/fchollet/keras/blob/master/keras/optimizers.py for more.



#Training:
	with sess.as_default():
		for dummy in range(train_count_):
			for cur_batch in range(batch_count):
				train_step.run(feed_dict={
					features: train_features[cur_batch*batch_size_:np.minimum((cur_batch+1)*batch_size_, train_labels.size), :].reshape( np.minimum((cur_batch+1)*batch_size_, train_labels.size) - cur_batch*batch_size_, 200),
					labels: train_labels[cur_batch*batch_size_: np.minimum((cur_batch+1)*batch_size_, train_labels.size)].reshape( np.minimum((cur_batch+1)*batch_size_, train_labels.size) - cur_batch*batch_size_, 1),
					K.learning_phase(): 1
				})
#Testing:
	
	#tf accuracy def:
	tf_correct_prediction = tf.equal(tf.round(output_layer), labels)
	tf_accuracy = tf.reduce_mean(tf.to_float(tf_correct_prediction))

	prediction = output_layer

	#keras accuracy def:
	#See: https://github.com/fchollet/keras/blob/master/keras/metrics.py -> not all metrics in docs
	from keras.metrics import binary_accuracy as k_acc
	k_accuracy = tf.reduce_mean(k_acc(labels, tf.round(output_layer)))
	acc = 0 #placeholder for accuracy.
	with sess.as_default():
		if verbose:
			print  sess.run(prediction, feed_dict={
					features: test_features,
					labels:np.asarray(test_labels).reshape(test_labels.size, 1),
					K.learning_phase(): 0
				})
		
		if using_keras_accuracy:
			acc = k_accuracy.eval(feed_dict={
				features: test_features,
				labels:np.asarray(test_labels).reshape(test_labels.size, 1),
				K.learning_phase(): 0
			})
		else:
			acc = sess.run(tf_accuracy, feed_dict={
				features: test_features,
				labels:np.asarray(test_labels).reshape(test_labels.size, 1),
				K.learning_phase(): 0
			})
	accuracy_info = "accuracy="+str(acc)
	print model_info+regulizer_info+", "+batch_info+", "+accuracy_info
	return acc
###end of predict function definition
import numpy as np
#predict( 0.1001, 0.03, 0.001, 0.03, 0.001, 100, 'sigmoid', 'sigmoid', "data/D1_S1.csv", "data/D1_S2.csv", False, True, 1) #Default configuration from solution 3.
#predict( 0.0002, 0.03, 0.001, 0.03, 0.001, 95, 'sigmoid', 'sigmoid', "data/D1_S1.csv", "data/D1_S2.csv", False, True, 2000) #No iteration and sticks close to 0.495 & 0.505, not good.
#batch size = 3 is not bad either.

#vals
learning_rate = 0.1
wl1 = 0.034
wl2 = 0.001
al1 = 0.334
al2 = 0.001
#to plot
x_i_max = 1
y_i_max = 3#5
x_ = np.zeros(x_i_max*y_i_max)
y_ = np.zeros(x_.size)#node
z_ = np.zeros(x_.size)#accuracy

#iteration
cur_pt = 0
for x in range(x_i_max):
	input_nodes = 92#92
	for y in range(y_i_max): #nodes
		x_[cur_pt] = learning_rate	#set x	
		y_[cur_pt] = input_nodes
		z_[cur_pt] = predict(learning_rate, wl1, wl2, al1, al2 ,
					input_nodes, 'sigmoid', 'sigmoid',
					"data/D1_S1.csv", "data/D1_S2.csv", False, True)
		input_nodes += 1
		cur_pt+=1
	learning_rate * 0.1
	
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlabel('?')
#ax.set_ylabel('Nodes')
#ax.set_zlabel('Test Accuracy')
#ax.scatter(x_, y_, z_, c='b')
#plt.show()








