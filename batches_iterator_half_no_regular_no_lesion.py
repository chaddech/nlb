import collections
import numpy as np
import scipy.misc
import os

class BatchesIterator(collections.Iterator):
	def __init__(self, batch_size, no_lesion_folders_path, 
									lesion_folders_path):

		self.batch_size = batch_size
		self.batch_start_index = 0
		self.files = []
		self.need_to_shuffle = True

		for folder in os.listdir(lesion_folders_path):
			#print 'lesion batches'
			folder_path = lesion_folders_path + folder + '/'
			for file_name in os.listdir(folder_path):
				self.files.append((folder_path+file_name, 1))

		for folder in os.listdir(no_lesion_folders_path):
			if ('crop1_train_p2' not in folder) and ('crop1_valid_p1' not in folder) and ('crop1_train_p3' not in folder) and ('crop1_train_p6' not in folder) and ('crop1_train_p1' not in folder) and ('crop1_train_p4' not in folder) and ('crop1_train_p5' not in folder):
				folder_path = no_lesion_folders_path + folder + '/'
				for file_name in os.listdir(folder_path):
					self.files.append((folder_path+file_name, 0))
					self.files.append((folder_path+file_name, 0))

		print len(self.files)

	def __iter__(self):
		return self

	def next(self):
		if self.need_to_shuffle:
			np.random.shuffle(self.files)
			self.need_to_shuffle = False
			self.batch_start_index = 0

		if len(self.files)-self.batch_start_index <= self.batch_size:
			self.need_to_shuffle = True
			this_batch_size = len(self.files) - self.batch_start_index
		else:
			this_batch_size = self.batch_size

		print "files index: ", self.batch_start_index
		batch_x = np.zeros((this_batch_size,3,224,224), dtype='uint8')
		
		batch_y = np.zeros((this_batch_size), dtype='uint8')

		for x in xrange(this_batch_size):
			pic = scipy.misc.imread(self.files[self.batch_start_index+x][0])
			batch_x[x]=pic.reshape(3,224,224)
			batch_y[x]=self.files[self.batch_start_index+x][1]
		self.batch_start_index+=self.batch_size
		return batch_x, batch_y



