
import numpy as np
import os
import scipy.misc
from scipy import ndimage
from batches_iterator34 import BatchesIterator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from scipy.misc import imread


from keras.callbacks import ModelCheckpoint
from keras import callbacks

model = Sequential()

directory = '/media/chad/nara/close_images/lesion2/heat_maps_valid/'

valid_no_lesion_dirs = ['tcrop13_no_lesion/', 'tcrop22-3_no_lesion940/', 'tcrop16_no_lesion937/']

valid_lesion_dirs = ['tcrop13_lesion/', 'tcrop22-3_lesion940/', 'tcrop16_lesion937/']


image_size=193*126
image_height = 126
image_width = 193
num_no_lesion_pics = 118
num_lesion_pics = 153
valid_no_lesion_pics=np.zeros((3,num_no_lesion_pics, image_height, image_width))
valid_lesion_pics=np.zeros((3,num_lesion_pics, image_height, image_width))

counter = 0
for filename in os.listdir(directory+valid_lesion_dirs[0]):
	direct_num = 0
	for direct in valid_lesion_dirs:
		pic = imread(directory+direct+filename)
		if pic.shape == (108,166):
			valid_lesion_pics[direct_num,counter,0:108, 0:166] = pic
		else:
			valid_lesion_pics[direct_num, counter, :,:]= pic
		direct_num += 1
	counter += 1
counter = 0

for filename in os.listdir(directory+valid_no_lesion_dirs[0]):
	direct_num = 0
	for direct in valid_no_lesion_dirs:
		pic = imread(directory+direct+filename)
		if pic.shape == (108,166):
			valid_no_lesion_pics[direct_num,counter,0:108, 0:166] = pic
		else:
			valid_no_lesion_pics[direct_num, counter, :,:]= pic
		direct_num += 1
	counter += 1

valid_no_lesion_pics = np.reshape(valid_no_lesion_pics, (num_no_lesion_pics, 3, image_height, image_width))
valid_lesion_pics = np.reshape(valid_lesion_pics, (num_lesion_pics, 3, image_height, image_width))

num_valid_pics = num_lesion_pics+num_no_lesion_pics
X_valid = np.zeros((num_valid_pics, 3, image_height, image_width))
Y_valid = np.zeros((num_valid_pics))

counter = 0
for m in valid_lesion_pics:
	X_valid[counter] = m
	counter += 1

Y_valid[0:counter] = 1
for m in valid_no_lesion_pics:
	X_valid[counter] = m
	counter += 1

train_no_lesion_dirs = ['tcrop13_no_lesion_train/', 'tcrop22-3_no_lesion940_train/', 'tcrop16_no_lesion937train/']

train_lesion_dirs = ['tcrop13_lesion_train/', 'tcrop22-3_lesion940_train/', 'tcrop16_lesion937train/']


num_no_lesion_pics_train = 532
num_lesion_pics_train = 721
train_no_lesion_pics=np.zeros((3,num_no_lesion_pics_train, image_height, image_width))
train_lesion_pics=np.zeros((3,num_lesion_pics_train, image_height, image_width))

counter = 0
for filename in os.listdir(directory+train_lesion_dirs[0]):
	direct_num = 0
	for direct in train_lesion_dirs:
		pic = imread(directory+direct+filename)
		if pic.shape == (108,166):
			train_lesion_pics[direct_num,counter,0:108, 0:166] = pic
		else:
			train_lesion_pics[direct_num, counter, :,:]= pic
		direct_num += 1
	counter += 1
counter = 0

for filename in os.listdir(directory+train_no_lesion_dirs[0]):
	direct_num = 0
	for direct in train_no_lesion_dirs:
		pic = imread(directory+direct+filename)
		if pic.shape == (108,166):
			train_no_lesion_pics[direct_num,counter,0:108, 0:166] = pic
		else:
			train_no_lesion_pics[direct_num, counter, :,:]= pic
		direct_num += 1
	counter += 1
train_no_lesion_pics = np.reshape(train_no_lesion_pics, (num_no_lesion_pics_train, 3, image_height, image_width))
train_lesion_pics = np.reshape(train_lesion_pics, (num_lesion_pics_train, 3, image_height, image_width))

num_training_pics = num_lesion_pics_train*4+num_no_lesion_pics_train*4
X_train = np.zeros((num_training_pics, 3, image_height, image_width))
Y_train = np.zeros((num_training_pics))

counter = 0

for m in train_lesion_pics:
	X_train[counter] = m
	counter += 1

	X_train[counter] = np.fliplr(np.copy(m))
	counter += 1

	X_train[counter] = np.flipud(np.copy(m))
	counter += 1

	X_train[counter] = np.rot90(np.copy(m),2)
	counter += 1



Y_train[0:counter] = 1

for m in train_no_lesion_pics:
	X_train[counter] = m
	counter += 1

	X_train[counter] = np.fliplr(np.copy(m))
	counter += 1

	X_train[counter] = np.flipud(np.copy(m))
	counter += 1

	X_train[counter] = np.rot90(np.copy(m),2)
	counter += 1



new_train_indices = np.arange(0, len(Y_train))
np.random.shuffle(new_train_indices)
X_train= X_train[new_train_indices,:,:,:]
Y_train= Y_train[new_train_indices]


model.add(Convolution2D(8,3,3,
			 init='he_normal', input_shape=(3,126,193), W_regularizer=l2(.0001)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))


model.add(Convolution2D(8,3,3, init='he_normal', W_regularizer=l2(.0001)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Convolution2D(8,3,3, init='he_normal', W_regularizer=l2(.0001)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,3,3, init='he_normal', W_regularizer=l2(.0001)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,3,3, init='he_normal', W_regularizer=l2(.0001)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,3,3, init='he_normal', W_regularizer=l2(.0001)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(8,3,3, init='he_normal', W_regularizer=l2(.0001)))
model.add(BatchNormalization(mode=0, axis=1))
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(1,init='glorot_normal'))

model.add(Activation('sigmoid'))
opti = Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opti, class_mode='binary')
model.summary()

file_name = 'stage_three_network'

saveWeigts = ModelCheckpoint(file_name+'_best_val_acc_weights.h5', monitor='val_acc', verbose=1, save_best_only=True)

cllbcks= [saveWeigts]

history_log = model.fit(X_train, Y_train, batch_size=64, nb_epoch=60, callbacks = cllbcks,
		validation_data=(X_valid,Y_valid),show_accuracy=True, verbose=1)

save_path='history/heatmaps/'+file_name+'/'
os.makedirs(save_path)
np.savetxt(save_path+file_name+'acc.txt',np.asarray(history_log.history['acc']), delimiter='\n')
np.savetxt(save_path+file_name+'loss.txt',np.asarray(history_log.history['loss']), delimiter='\n')
np.savetxt(save_path+file_name+'val_loss.txt',np.asarray(history_log.history['val_loss']), delimiter='\n')
np.savetxt(save_path+file_name+'val_acc.txt',np.asarray(history_log.history['val_acc']), delimiter='\n')
