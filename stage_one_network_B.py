
import numpy as np
import os
import scipy.misc
from scipy import ndimage
from batches_iterator3 import BatchesIterator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from IPython.display import SVG

from keras.callbacks import ModelCheckpoint
from keras import callbacks

model = Sequential()

valid_lesion_dir = '/media/chad/delft/crop1/lesion/crop1_all_valid_folders/'
valid_no_lesion_dir = '/media/chad/delft/crop1/no_lesion/crop1_all_valid_folders/'
train_lesion_dir = '/media/chad/delft/crop1/lesion/crop1_all_train_folders/'
train_no_lesion_dir = '/media/chad/delft/crop1/no_lesion/crop1_all_train_folders/'

train_batch_size = 40000
valid_batch_size = 50000

train_batches = BatchesIterator(train_batch_size,train_no_lesion_dir,train_lesion_dir)
valid_batches = BatchesIterator(valid_batch_size,valid_no_lesion_dir,valid_lesion_dir)
X_valid, Y_valid = valid_batches.next()





model.add(BatchNormalization(mode=0, axis=1, input_shape=(3,224,224)))
model.add(Convolution2D(64,3,3,
			 init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))


model.add(Convolution2D(64,3,3,init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(128,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(256,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(256,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(512,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(512,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(32,3,3, init='he_normal'))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(1,init='glorot_normal'))

model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')
model.summary()

file_name = 'network_B_1'
saveWeigts = ModelCheckpoint(file_name+'_best_weights.h5', monitor='val_acc', verbose=1, save_best_only=True)

cllbcks= [saveWeigts]



for mini_epoch in range(1000):
	print "Epoch ", mini_epoch

	X_train, Y_train = train_batches.next()

	model.fit(X_train, Y_train, batch_size=64, nb_epoch=1, callbacks = cllbcks,
			validation_data=(X_valid,Y_valid),show_accuracy=True, verbose=1)

