from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv1D, Conv2D, Conv3D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, MaxPooling1D, AveragePooling2D, MaxPooling3D, Flatten, Dropout, Lambda, Layer, Input, add as Kadd, Bidirectional,GRU
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from os.path import join
import keras.backend as K
K.set_image_data_format('channels_last') # set format
import tensorflow as tf
import numpy as np

def lrcn(input_shape, nb_classes,train=True):
	"""Build a CNN into RNN.
	Starting version from:
		https://github.com/udacity/self-driving-car/blob/master/
			steering-models/community-models/chauffeur/models.py
	Heavily influenced by VGG-16:
		https://arxiv.org/abs/1409.1556
	Also known as an LRCN:
		https://arxiv.org/pdf/1411.4389.pdf
	"""
	model = Sequential()

	model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), activation='relu', padding='same'), input_shape=input_shape))
	model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer="he_normal", activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
	
	model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
	model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

	model.add(TimeDistributed(Flatten()))

	model.add(Dropout(0.5))
	model.add(LSTM(256, return_sequences=False, dropout=0.5))
	
	if not train:
		model.load_weights(join('Testdata3','checkpoints','LRCN.003-6.131.hdf5'), by_name=True)
		
	model.add(Dense(nb_classes, activation='softmax'))

	return model

def TC_5(input_shape, nb_classes, train=False):
	model = Sequential(
		[ ##conv1
		Conv3D(96,(5,7,7), activation = 'relu', input_shape=input_shape),
		#Activation('relu'),
		#LRN
		#Lambda(lrn),
		## pool1
		MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2)),
		## conv2
		Conv3D(256,(1,5,5), activation = 'relu'),
		#Activation('relu'),
		#LRN
		#Lambda(lrn),
		## pool2
		MaxPooling3D(pool_size=(1,3,3),strides=(1,2,2)),
		## conv3
		Conv3D(256,(1,3,3),activation='relu'),
		#ZeroPadding2D((1,1)),
		## conv4
		Conv3D(256,(1,3,3),activation='relu'),
		#ZeroPadding2D((1,1)),
		## conv5
		Conv3D(256,(1,3,3),activation='relu'),
		## pool5
		MaxPooling3D(pool_size=(1,3,3),strides=(1,2,2)),
		## conv6
		Conv3D(512,(1,6,6),activation='relu'),
		## conv7
		Conv3D(1,(5,1,1),activation='relu'),
		## conv8
		Conv3D(1,(1,1,1),activation='relu'),
		Flatten()
		]
	)
	model.load_weights(join('Testdata3','checkpoints','TC5.003-6.131.hdf5'), by_name=True)
	model.add(Dense(nb_classes, activation='softmax'))
	return model

def TC_5(input_shape, nb_classes, train):
	'''W = np.load('model.npy').item()
	def conv(num):
		c = W['conv{}'.format(num)]
		return c['weights'], c['biases']

	def fc(num):
		c = W['fc{}'.format(num)]
		return c['weights'], c['biases']'''

	model = Sequential(
		[ ##conv1
		Conv3D(96,(5,7,7), activation = 'relu', input_shape=input_shape),
		#Activation('relu'),
		#LRN
		#Lambda(lrn),
		## pool1
		MaxPooling3D(pool_size=(1,3,3), strides=(1,2,2)),
		## conv2
		Conv3D(256,(1,5,5), activation = 'relu'),
		#Activation('relu'),
		#LRN
		#Lambda(lrn),
		## pool2
		MaxPooling3D(pool_size=(1,3,3),strides=(1,2,2)),
		## conv3
		Conv3D(256,(1,3,3),activation='relu'),
		#ZeroPadding2D((1,1)),
		## conv4
		Conv3D(256,(1,3,3),activation='relu'),
		#ZeroPadding2D((1,1)),
		## conv5
		Conv3D(256,(1,3,3),activation='relu'),
		## pool5
		MaxPooling3D(pool_size=(1,3,3),strides=(1,2,2)),
		## conv6
		Conv3D(512,(1,6,6),activation='relu'),
		## conv7
		Conv3D(1,(5,1,1),activation='relu'),
		## conv8
		Conv3D(1,(1,1,1),activation='relu'),
		Flatten(),
		## fc9
		Dense(nb_classes,activation='softmax')
		]
	)
	return model

def lstm(input_shape,nb_classes,train=True):
	"""Build a simple LSTM network. We pass the extracted features from
	our CNN to this model predomenently."""
	# Model.
	model = Sequential()
	model.add(LSTM(2048, return_sequences=False,
				   input_shape=input_shape,
				   dropout=0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	if not train:
		model.load_weights(join('Testdata3','checkpoints','lstm.011-2.427.hdf5'), by_name=True)
	model.add(Dense(nb_classes, activation='softmax'))

	return model
'''	
class Lipreading(keras.Model):
	def __init__(self, mode, input_shape=(29,224,224,3),nb_classes=10):
		self.mode = mode
		self.input_shape = input_shape
		self.nb_classes = nb_classes
		
	def call:
'''
def getTranspose(x):
	if len(K.int_shape(x)) == 5:
		return K.permute_dimensions(x, (0,2,3,1,4))
	else: 
		return K.permute_dimensions(x, (1,0))
	
def outputTrans(input_shape):
	if len(input_shape) == 2:
		return(input_shape[1],input_shape[0])
	elif len(input_shape) == 3:
		return (input_shape[0],input_shape[2],input_shape[1])
	else:
		return (input_shape[0],input_shape[2],input_shape[3],input_shape[1],input_shape[4])

def BasicBlock(inplanes,planes,namestr,stride=1,downsample=None):
	input = Input(shape=(24,24,inplanes))
	residual = input
	x = Conv2D(planes,kernel_size=3,strides=stride,padding='same',use_bias=False,name=namestr+'_Conv2D')(input)
	##x = BatchNormalization(name=namestr+'_BN')(x)
	x = Activation('relu',name=namestr+'_act')(x)
	out = Conv2D(planes,kernel_size=3,strides=1,padding='same',use_bias=False)(x)
	##out = BatchNormalization(name=namestr+'_BN2')(conv2)
	if downsample is not None:
		residual = downsample(input)
	out = Kadd([out,residual])
	out = Activation('relu',name=namestr+'_act2')(out)
	model = Model(inputs = input, outputs = out)
	return model 
	
inplanes = 64
def make_layer(planes,blocks, namestr, stride=1):
	global inplanes
	downsample = None
	if stride != 1 or planes != 64:
		downsample = Sequential([
			Conv2D(planes,kernel_size=1,strides=stride,use_bias=False,name=namestr+'_Conv2D'),
			##BatchNormalization(name=namestr+'_BN')
		])
	layers = []
	layers.append(BasicBlock(inplanes,planes, namestr+'_'+str(0),stride, downsample))
	inplanes = planes
	for i in range(1,blocks):
		layers.append(BasicBlock(inplanes,planes,namestr+'_'+str(i)))
	return Sequential(layers,name=namestr)
	
def ResNet(layers,num_classes,input):
	global inplanes
	layer1 = make_layer(64,layers[0])
	layer2 = make_layer(128,layers[1],stride=2)
	layer3 = make_layer(256,layers[2],stride=2)
	layer4 = make_layer(512,layers[3],stride=2)
	x = layer1(input)
	x = layer2(x)
	x = layer3(x)
	x = layer4(x)
	x = AveragePooling2D(pool_size=(2,2))(x)
	x = Flatten()(x)
	x = Dense(num_classes)(x)
	x = BatchNormalization()(x)
	return x
	
def takeMean(x):
	return K.mean(x,1)
def mean_out(input_shape):
	return (input_shape[0],input_shape[2])
	
def reshape(x):
	global batchsize
	sh = K.int_shape(x)
	if len(sh) == 5:
		return K.reshape(x,shape=tf.convert_to_tensor([sh[1]*batchsize,sh[2],sh[3],sh[4]]))
	else:
		return K.reshape(x,shape=tf.convert_to_tensor([batchsize,int(sh[0]/batchsize),sh[1]]))
def out_reshape(input_shape):
	global batchsize
	if len(input_shape)==5:
		return (input_shape[1]*batchsize,input_shape[2],input_shape[3],input_shape[4])
	else:
		return (batchsize,int(input_shape[0]/batchsize),input_shape[1])

batchsize = 15
def lipreading(mode, inputDim=256, hiddenDim =512, input_shape=(29,88,88,1),nb_classes=12,train=True,file=None):
	global batchsize
	input = Input(shape=input_shape, name = 'input')
	print (K.int_shape(input))
	
	# frontend 3D
	x = Conv3D(64,(5,7,7),strides=(1,2,2),padding='same', use_bias=False,name = 'fe_conv3d_1')(input)
	print (K.int_shape(x))
	x = BatchNormalization(name='fe_bn1')(x)
	print (K.int_shape(x))
	x = Activation('relu',name='fe_act1')(x)
	print (K.int_shape(x))
	x = MaxPooling3D(pool_size=(1,3,3),strides=(1,2,2),padding='same',name='fe_maxpool1')(x)
	print (K.int_shape(x))
	x = Lambda(reshape,output_shape=out_reshape,name='reshape1')(x)
	print (K.int_shape(x))
	
	# resnet 34
	layers = [3,4,6,3]
	layer1 = make_layer(64,layers[0],'layer1')
	layer2 = make_layer(128,layers[1],'layer2',stride=2)
	layer3 = make_layer(256,layers[2],'layer3',stride=2)
	layer4 = make_layer(512,layers[3],'layer4',stride=2)
	x = layer1(x)
	x = layer2(x)
	x = layer3(x)
	x = layer4(x)
	x = AveragePooling2D(pool_size=(2,2),name = 'resnet_avgpool')(x)
	x = Flatten(name='resnet_flatten_1')(x)
	x = Dense(inputDim,name='resnet_dense1')(x)
	x = BatchNormalization(name='resnet_BN')(x)
	print (K.int_shape(x))

	x = Lambda(reshape,output_shape=out_reshape,name='reshape2')(x)
	print (K.int_shape(x))
	
	model = Model(inputs=input, outputs = x)
	print('num layers: ',len(model.layers))
	
	backend_conv1 = Sequential([
		Conv1D(2*inputDim,5, strides=2,use_bias=False,name='bc1_conv1d_1'),
		##BatchNormalization(name='bc1_BN'),
		Activation('relu',name='bc1_act1'),
		MaxPooling1D(pool_size=2,strides=2,name='bc1_maxpool'),
		Conv1D(4*inputDim,5,strides =2, use_bias=False,name='bc1_conv1d_2'),
		##BatchNormalization(name='bc1_BN2'),
		Activation('relu',name='bc1_act2')
		],name='backend_conv1')
	
	backend_conv2 = Sequential([
		Dense(inputDim,name='bc2_dense_1'),
		##BatchNormalization(name='bc2_BN'),
		Activation('relu',name='bc2_act'),
		Dense(nb_classes, activation='softmax',name='bc2_softmax')
		],name='backend_conv2')
	
	# gru
	gru = Sequential([
		Bidirectional(GRU(hiddenDim,recurrent_activation='sigmoid', return_sequences = True,reset_after=True,name='gru1')),
		Bidirectional(GRU(hiddenDim,recurrent_activation='sigmoid', reset_after=True,name='gru2')),
		Dense(nb_classes,activation='softmax',name='gru_softmax')
		],name='gru')
	
	if mode == 'temporalConv':
		x = backend_conv1(x)
		print (K.int_shape(x))
		x = Lambda(takeMean,output_shape=mean_out,name='mean_layer')(x)
		print (K.int_shape(x))
		x = backend_conv2(x)
		print (K.int_shape(x))
		
		model_end = Model(inputs=input, outputs = x)
	elif mode == 'backendGRU' or mode == 'finetuneGRU':
		x = gru(x)
		model_end = Model(inputs=input, outputs = x)
		
	elif mode == 'backendLSTM':
		x = LSTM(2048, return_sequences=False,dropout=0.5,name='LSTM')(x)
		x = LSTM(2048, return_sequences=False,dropout=0.5,name='LSTM')(x)
		x = Dense(512, activation='relu',name='lstm_relu')(x)
		x = Dropout(0.5,name = 'dropout')(x)
		x = Dense(nb_classes,activation='softmax',name='lstm_softmax')(x)
		model_end = Model(inputs=input, outputs= x)
		if not train:
			model_end.load_weights(file, by_name=True)
			for layer in model_end.layers[:15]:
				layer.trainable = False
	else:
		raise Exception('No model selected')
		
	model_end.summary()
	return model_end
