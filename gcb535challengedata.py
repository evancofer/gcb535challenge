#Original idea from: https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py , with some minor improvements/changes.

from __future__ import absolute_import
from __future__ import with_statement
from __future__ import print_function

import os
import pysam as ps
import collections
import numpy as np
import itertools as it

from six.moves import urllib
DEFAULT_VALIDATION_FRACTION = 0.2
SOURCE_URL = 'https://raw.githubusercontent.com/greenelab/gcb535challenge/master/'

def maybe_download(filename, subdirectory, work_directory='', local_only=False):
	if not(os.path.exists(os.path.join(work_directory, subdirectory))):
		if local_only:
			raise IOError('Could not find locally:'+str(os.path.join(work_directory, subdirectory)))
		else:
			os.mkdir(os.path.join(work_directory, subdirectory))
	fpath = os.path.join(work_directory, subdirectory)
	fpath = os.path.join(fpath, filename)
	if not(os.path.exists(fpath)):
		if local_only:
			raise IOError('Could not find locally:'+str(fpath))
		else:
			fpath, _ = urllib.request.urlretrieve(SOURCE_URL + subdirectory + '/' + filename, fpath)
			print('Downloaded ',filename,', ', os.stat(fpath).st_size,' bytes')
	return fpath



def extract_dataset(filename, label_as_pair=False):
	data = np.loadtxt(open(filename, "rb"), delimiter=",")
	if label_as_pair:
		return (data[:, :200], map(lambda x: np.array([1, 0]) if x == 0 else np.array([0, 1]), data[:, 200]))
	else:
		return (data[:, :200], data[:, 200].reshape(len(data), 1))
	


class DataSet(object):


	def __init__(self, dataset, label_as_pair=False):

		self._label_as_pair = label_as_pair
		self._num_points = len(dataset[0])
		self._labels = dataset[1]
		self._features = dataset[0]
		
		self._epochs_completed = 0
		self._index_in_epoch = 0


	@property
	def features(self):
		return self._features


	@property
	def labels(self):
		return self._labels


	@property
	def size(self):
		return self._num_points

	@property
	def epochs_completed(self):
		return self._epochs_completed


	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch >= self._num_points:
			self._epochs_completed += 1
			#Shuffle data
			permut = np.arange(self._num_points)
			np.random.shuffle(permut)
			self._features = self._features[permut]
			self._labels = self._labels[permut]
			#Adjust start
			start = 0
			self._index_in_epoch = batch_size
			assert (batch_size < self._num_points), (
				"This set's batch size should not exceed:" + str(self._num_points))
		end = self._index_in_epoch
		return self._features[start:end], self._labels[start:end]



def read_data_sets(
	dataset_index, 
	train_indices=np.array([1,2,3]), 
	test_indices=np.array([4]),  
	validation_fraction=DEFAULT_VALIDATION_FRACTION,
	work_directory = '',
	local_only=False,
	label_as_pair=False):


	class DataSets(object):
		pass
	data_sets=DataSets()

	train_indices = np.unique(np.asarray(train_indices))
	test_indices = np.unique(np.asarray(test_indices))


	assert (dataset_index == 1 or dataset_index == 2), (
		'Not a valid dataset index. Try again with 1 or 2.')
	assert (np.intersect1d(train_indices, test_indices).size == 0), (
		'Your training set and testing set should not overlap!')
	assert train_indices.size > 0, ('You currently have no elements in your training set.')
	assert test_indices.size > 0, ('You currently have no elements in your testing set.')
	assert all( 0 < x < 5 for x in train_indices), "You have a training index other than 1, 2, 3, and 4."
	assert all( 0 < x < 5 for x in test_indices), "You have a testing index other than 1, 2, 3, and 4."
	assert validation_fraction < 1.0, "Validation fraction cannot be 1"


	filenames = lambda x: map(lambda y: ('D%s_S%s.csv' %(dataset_index, y), 'predict' if y == 4 else 'data'), x)
	TRAIN_FILES = filenames(train_indices)
	TEST_FILES = filenames(test_indices)


	fetch_files = lambda x: map(lambda y: maybe_download(y[0],  y[1], work_directory, local_only), x)
	TRAIN_FILES = fetch_files(TRAIN_FILES)
	TEST_FILES = fetch_files(TEST_FILES)

	
	form_dataset = lambda x: map(lambda r: np.concatenate(list(r), 0), map(it.chain, zip(*map(lambda y: extract_dataset(y, label_as_pair), x)) ))
	train_data = form_dataset(TRAIN_FILES)
	test_data = form_dataset(TEST_FILES)
	

	validation_size = int(len(train_data[1]) * validation_fraction)
	validation_data = (train_data[0][:validation_size], train_data[1][:validation_size])
	train_data = (train_data[0][validation_size:], train_data[1][validation_size:])

	data_sets.train = DataSet(train_data) 
	"""data_sets.validation = DataSet(validation_data)"""
	data_sets.test = DataSet(test_data)

	return data_sets
