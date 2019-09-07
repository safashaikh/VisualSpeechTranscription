import imageio
import json
import numpy as np
import sys
import cv2
import os
from datetime import date
from imutils import face_utils
import imutils
import dlib
from collections import OrderedDict
from models import lipreading, batchsize
from keras.models import model_from_json, load_model
import keras_metrics

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
	
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

def key_to_stop(c):
	while True:
	# display the image and wait for a keypress
		key = cv2.waitKey(1) & 0xFF
	# if the 'c' key is pressed, break from the loop
		if key == ord(c):
			break
			
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

def rescale_list(input_list, size):
	"""Given a list and a size, return a rescaled/samples list. For example,
	if we want a list of size 5 and we have a list of size 25, return a new
	list of size five which is every 5th element of the origina list."""
	assert len(input_list) >= size

	# Get the number to skip between iterations.
	skip = len(input_list) // size
	
	#print("skip is: ", skip)
	if (skip ==1):
		start = (len(input_list) - size)// 2 
		end = start + size
		output = input_list[start:end]
	else:
	# Build our new output.
		output = [input_list[i] for i in range(0, len(input_list), skip)]
	#print("len output is: ",len(output))
	# Cut off the last one if needed.
	return output[:size]
	
def rescale_frame(frame, percent=100):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def load_frames(path):
	# depending on your environment, this might sometimes produce 30 frames
	cap = np.array(list(imageio.get_reader(path, 'ffmpeg')))
	#print(cap.shape[0])
	cap = rescale_list(cap,29)
	#for x,y,w,h in mouth:
	u_images = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_BGR2GRAY) for _ in range(29)], axis=0)
	
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('landmarks.dat')
	#print(u_images.shape)
	
	# detect faces in the grayscale image
	rects = detector(u_images[0], 1)
	rect = rects[0]
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(u_images[0], rect)
	shape = face_utils.shape_to_np(shape)
	(j, k) = FACIAL_LANDMARKS_IDXS['mouth']
	pts = shape[j:k]
	#mean = np.mean(pts,axis=0)
	min = np.min(pts,axis=0)
	max = np.max(pts,axis=0)
	avg = max - min
	#print("width is: ",avg)
	pc = 43/avg[0] * 100
	
	images = []
	for i in range(len(u_images)):
		#mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
	
		r_img = rescale_frame(u_images[i],pc)
		rects = detector(r_img, 1)
		rect = rects[0]
		shape = predictor(r_img, rect)
		shape = face_utils.shape_to_np(shape)
		(j, k) = FACIAL_LANDMARKS_IDXS['mouth']
		pts = shape[j:k]
		mean = np.mean(pts,axis=0)
		center_x = int(mean[0])
		center_y = int(mean[1])
		c_img = r_img[center_y-44:center_y+44,center_x-44:center_x+44]
		images.append(c_img)
		#cv2.imshow('img1',r_img)
		cv2.imshow('img',c_img)
		cv2.waitKey(100)
		#key_to_stop('n')
	#images = images[:, 115:211, 79:175] / 255.
	#print(np.array(images).shape)
	images = np.array(images)#/255.
	#print(images.shape)
	#mean = 0.413621
	#std = 0.1700239
	#images = (images - mean) / std
	images = images.reshape(29, 88, 88, 1)
	list15 = []
	for i in range(15):
		list15.append(images)
	images = np.array(list15)
	#print("images shape is ", images.shape)
	#print(images.shape)
	#images = torch.tensor(images, dtype=torch.float32)
	return images

classes = {}
file = open('classes12.txt','r')
i = 0
for line in file:
    classes[i] = line.split('\n')[0]
    i+=1
file.close()
    
# load model
'''import keras.backend as K
import tensorflow as tf
model = load_model('saved_models_lip_model.h5',custom_objects = {'tf':tf,'K':K,'batchsize':batchsize})
'''

# load json and create model
'''json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")'''

model = lipreading('temporalConv')
model.load_weights('lipreading_weights.021-0.862.hdf5')
#model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics = ['acc',keras_metrics.precision(), keras_metrics.recall(),keras_metrics.f1_score(),keras_metrics.false_positive(), keras_metrics.false_negative(), keras_metrics.true_negative(),keras_metrics.true_positive()])

if os.path.exists(os.path.join('results','result' + '.json')):
	with open(os.path.join('results','result' + '.json'),"r") as f:
		data = json.load(f)
	if "resp" not in data.keys():
		data["resp"] = []
else:
	data = {}


#for j in range(2, 3):
images = load_frames(sys.argv[1])
 
# print the label
output = model.predict(images)[0]
#print(output)
top_class_idx = np.argmax(output)
top_class_label = classes[top_class_idx]
sorted_indices = np.argsort(output)[::-1]
sorted_labels = []
sorted_probabilities = []
output_dict = OrderedDict()
for i in range(len(sorted_indices)):
	sorted_labels.append(classes[sorted_indices[i]])
	sorted_probabilities.append(output[sorted_indices[i]])
	label = classes[sorted_indices[i]]
	prob = float(output[sorted_indices[i]])
	output_dict[label] = prob
print(sorted_labels)
print(sorted_probabilities)

data['resp'] = []

data['resp'].append({
	'date': str(date.today()),
	'method': 'video',
	'result': top_class_label,
	'all_results': output_dict
	})

#with open(os.path.join('results',os.path.basename(sys.argv[1]).split('.')[0] + '.json'),'w') as fp:
with open(os.path.join('results','result' + '.json'),'w') as fp:
    json.dump(data,fp)
