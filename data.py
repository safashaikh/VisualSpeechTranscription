import csv
import numpy as np
import random
import glob
import os.path
import sys
import operator
import threading
from keras.utils import to_categorical
import matplotlib.image as img
import cv2

class threadsafe_iterator:
	def __init__(self, iterator):
	    self.iterator = iterator
	    self.lock = threading.Lock()

	def __iter__(self):
	    return self

	def __next__(self):
	    with self.lock:
	        return next(self.iterator)

def threadsafe_generator(func):
	"""Decorator"""
	def gen(*a, **kw):
	    return threadsafe_iterator(func(*a, **kw))
	return gen

class DataSet():
	def __init__(self, seq_length = 10, class_limit = None, image_shape=(96,96,3), is1d = False):

		self.seq_length = seq_length
		self.class_limit = class_limit
		# ?self.sequence_path = os.path.join('data', 'sequences')
		self.max_frames = 200

		self.data = self.get_data()

		self.classes = self.get_classes()

		self.data = self.clean_data()

		self.image_shape = image_shape
		self.sequence_path = os.path.join('Testdata3','sequences')
		self.is1d = is1d

	@staticmethod
	def get_data():
		with open(os.path.join('Testdata3', 'data_file.csv'), 'r') as fin:
			reader = csv.reader(fin)
			data = list(reader)
		return data

	def clean_data(self):
		data_clean = []
		for item in self.data:
			if int(item[3]) >=self.seq_length and int(item[3]) <= self.max_frames and item[1] in self.classes:
				data_clean.append(item)
		return data_clean

	def get_classes(self):
		data_classes = []
		for item in self.data:
			if item[1] not in data_classes:
				data_classes.append(item[1])
		data_classes = sorted(data_classes)
		if self.class_limit is not None:
			return data_classes[:self.class_limit]
		else:
			return data_classes

	def split_train_test(self):
		train = []
		test = []
		for item in self.data:
			if item[0] == 'train':
				train.append(item)
			else:
				test.append(item)
		return train, test
	def get_class_one_hot(self, class_str):
		label_encoded = self.classes.index(class_str)

		# Now one-hot it.
		label_hot = to_categorical(label_encoded, len(self.classes))

		assert len(label_hot) == len(self.classes)
		return label_hot
	
	@threadsafe_generator
	def frame_generator(self, batch_size, train_test, data_type):
		train, test = self.split_train_test()
		data = train if train_test == 'train' else test
		print("Creating %s generator with %d samples." % (train_test, len(data)))
		while 1:
			X, y = [], []
			for i in range(batch_size):
				sequence = None
				sample = random.choice(data)
				if data_type is "images":
					# Get and resample frames.
					frames = self.get_frames_for_sample(sample)
					#frames = self.rescale_list(frames, self.seq_length)
					sequence = self.build_image_sequence(frames)
					#print(sequence.size())
					sequence = np.array(sequence)
				else:
					# Get the sequence from disk.
					sequence = self.get_extracted_sequence(data_type, sample)

					if sequence is None:
					    raise ValueError("Can't find sequence. Did you generate them?")
					# Get a random sample.
				
				X.append(sequence)
				y.append(self.get_class_one_hot(sample[1]))

			X = np.array(X)
			yield X, np.array(y)
	def get_extracted_sequence(self, data_type, sample):
		"""Get the saved extracted features."""
		filename = sample[2]
		path = os.path.join(self.sequence_path, filename + '-' + str(self.seq_length) + \
		    '-' + data_type + '.npy')
		if os.path.isfile(path):
			return np.load(path)
		else:
			return None	
	def build_image_sequence(self, frames):
		#mouth_cascade = cv2.CascadeClassifier('/Users/timmliu16888/Downloads/haarcascade_mcs_mouth.xml')
		sequence = list()
		for x in frames:
			img = cv2.imread(x)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gray = gray[4:92, 4:92, np.newaxis]
			if (self.is1d):
				sequence.append(gray)#sequence.append(np.reshape(gray,(gray.shape[0],gray.shape[1],1)))
			else: 
				sequence.append(img)
		'''sequence = np.array(sequence)/255
		mean = 0.413621
		std = 0.1700239
		sequence = (sequence - mean) / std'''
		return sequence

	@staticmethod
	def rescale_list(input_list, size):
		"""Given a list and a size, return a rescaled/samples list. For example,
		if we want a list of size 5 and we have a list of size 25, return a new
		list of size five which is every 5th element of the origina list."""
		assert len(input_list) >= size

		# Get the number to skip between iterations.
		skip = len(input_list) // size

		# Build our new output.
		output = [input_list[i] for i in range(0, len(input_list), skip)]

		# Cut off the last one if needed.
		return output[:size]	
	@staticmethod
	def get_frames_for_sample(sample):
		path = os.path.join('Testdata3', sample[0], sample[1])
		filename = sample[2]
		images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
		return images




