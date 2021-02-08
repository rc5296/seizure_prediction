import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# import keras
import tensorflow.keras as keras
from keras.models import Sequential, Model
from keras.layers import add, concatenate, Input
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D
from keras.layers.recurrent import GRU, LSTM
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adagrad,Adadelta,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import max_norm
from keras import backend as K

from models.customCallbacks import MyEarlyStopping, MyModelCheckpoint

class ConvNN(object):
	def __init__(self,target,batch_size=16,nb_classes=2,epochs=2,mode='cv'):
		self.target = target
		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.epochs = epochs
		self.mode = mode

	def setup(self,X_train_shape):
		print ('X_train shape', X_train_shape)
		# Input shape = (None,1,22,59,114)
		inputs = Input(shape=X_train_shape[1:])

		normal1 = BatchNormalization(
			# axis=2,
			axis=1,
			name='normal1')(inputs)

		# conv1 = Conv3D(
		# 	16,
		# 	kernel_size=(X_train_shape[2], 5, 5),
		# 	padding='valid',strides=(1,2,2),
		# 	name='conv1')(normal1)
		conv1 = Convolution2D(16, (5, 5), padding='valid', strides=(2,2), name='conv1')(normal1)

		relu1 = Activation('relu')(conv1)

		# pool1 = MaxPooling3D(
		# 	pool_size=(1,2,2),
		# 	padding='same')(relu1)
		pool1 = MaxPooling2D(pool_size=(2,2))(relu1)

		# ts = int(np.round(np.floor((X_train_shape[3]-4)/2) / 2 + 0.1))
		# fs = int(np.round(np.floor((X_train_shape[4]-4)/2) / 2 + 0.1))

		# reshape1 = Reshape((16,ts,fs))(pool1)
		# o_shape = pool1.shape
		# reshape1 = Reshape((o_shape[1], o_shape[3], o_shape[4]))(pool1)

		# normal2 = BatchNormalization(axis=1, name='normal2')(reshape1)
		normal2 = BatchNormalization(axis=1, name='normal2')(pool1)

		conv2 = Convolution2D(32, (3, 3),
							  padding='same',strides=(1,1), name='conv2')(normal2)
		relu2 = Activation('relu')(conv2)
		pool2 = MaxPooling2D(pool_size=(2,2))(relu2)

		normal3 = BatchNormalization(axis=1, name='normal3')(pool2)

		conv3 = Convolution2D(64, (3, 3),
							  padding='same',strides=(1,1), name='conv3')(normal3)
		relu3 = Activation('relu')(conv3)
		pool3 = MaxPooling2D(pool_size=(2,2))(relu3)

		flat = Flatten()(pool3)

		drop1 = Dropout(0.5)(flat)

		dens1 = Dense(128, activation='sigmoid', name='dens1')(drop1)
		drop2 = Dropout(0.5)(dens1)

		dens2 = Dense(self.nb_classes, name='dens2')(drop2)

		# option to include temperature in softmax
		temp = 1.0
		temperature = Lambda(lambda x: x / temp)(dens2)
		last = Activation('softmax')(temperature)

		self.model = Model(inputs=inputs, outputs=last)

		adam = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
		self.model.compile(loss='categorical_crossentropy',
					  optimizer=adam,
					  metrics=['accuracy'])

		print (self.model.summary())
		return self


	def fit(self,X_train,Y_train,X_val=None, y_val=None):
		Y_train = Y_train.astype('uint8')
		Y_train = np_utils.to_categorical(Y_train, self.nb_classes)
		y_val = np_utils.to_categorical(y_val, self.nb_classes)

		# early_stop = MyEarlyStopping(patience=10, verbose=0)
		early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
		checkpointer = MyModelCheckpoint(
			filepath="weights_%s_%s.h5" %(self.target, self.mode),
			verbose=0, save_best_only=True)


		if (y_val is None):
			self.model.fit(X_train, Y_train, batch_size=self.batch_size,
						   epochs=self.epochs,validation_split=0.2,
						   callbacks=[early_stop,checkpointer], verbose=2
						   )
		else:
			self.model.fit(X_train, Y_train, batch_size=self.batch_size,
						   epochs=self.epochs,validation_data=(X_val,y_val),
						   callbacks=[early_stop,checkpointer], verbose=2
						   )
		self.model.load_weights("weights_%s_%s.h5" %(self.target, self.mode))
		if self.mode == 'cv':
			os.remove("weights_%s_%s.h5" %(self.target, self.mode))
		return self

	def load_trained_weights(self, filename):
		self.model.load_weights(filename)
		print ('Loading pre-trained weights from %s.' %filename)
		return self

	def predict_proba(self,X):
		return self.model.predict([X])

	def evaluate(self, X, y):
		predictions = self.model.predict(X, verbose=0)

		auc_test = metrics.roc_auc_score(y, predictions[:,1])
		# fpr, tpr, threshold = metrics.roc_curve(y, predictions[:,1])
		# roc_auc = metrics.auc(fpr, tpr)
		# plt.title('Receiver Operating Characteristic (Patient 1, Seizure   )')
		# plt.plot(fpr, tpr, 'b', label='AUC = {:.2f}'.format(roc_auc))
		# plt.legend(loc='lower right')
		# plt.plot([0,1], [0,1], 'r--')
		# plt.xlim([0,1])
		# plt.ylim([0,1])
		# plt.ylabel('True Positive Rate')
		# plt.xlabel('False Positive Rate')
		# plt.show()
		print('Test AUC is:{}'.format(auc_test))

		class_predict = np.argmax(predictions, axis=1)
		pos_idx = np.nonzero(y == 1)
		neg_idx = np.nonzero(y == 0)

		threshold = (6,8)
		true_pos = getTruePos(class_predict[pos_idx], y[pos_idx], threshold)
		false_pos = getFalsePos(class_predict[neg_idx], y[neg_idx], threshold)

		return auc_test, true_pos, false_pos, class_predict

def getTruePos(predictions, actual, threshold):
	"""

	:param predictions:
	:param actual:
	:param threshold: Tuple (k,n) for "k-of-n" method of determining alarm
	"""
	count = 0
	j = 0
	nArr = []

	for i in range(len(predictions)):
		if count < threshold[1]:
			nArr.append(predictions[i])
			count += 1
			if sum(nArr) >= threshold[0]: # sound alarm
				return 1

		else:
			nArr[j] = predictions[i]
			if sum(nArr) >= threshold[0]: # sound alarm
				return 1
			j = (j + 1) % threshold[1]

	return 0

def getFalsePos(predictions, actual, threshold):
	alarm = False
	disabledClips = 69 # amount of time the alarm lasts for, set to n where time=n*clipLength - 1
	disableCount = 0
	fpos = 0
	count = 0
	j = 0
	nArr = []

	for i in range(len(predictions)):
		if not alarm:
			if count < threshold[1]:
				nArr.append(predictions[i])
				count += 1
				if sum(nArr) >= threshold[0]:  # sound alarm
					fpos += 1
					alarm = True

			else:
				nArr[j] = predictions[i]
				if sum(nArr) >= threshold[0]:  # sound alarm
					fpos += 1
					alarm = True

				j = (j + 1) % threshold[1]

		elif disableCount == disabledClips:
			disableCount = 0
			count = 0
			nArr = []
			j = 0
			alarm = False

		else:
			disableCount += 1

	return fpos