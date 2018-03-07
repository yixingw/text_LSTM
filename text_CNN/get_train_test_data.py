import numpy as np 
import copy

def get_data(train_data_path,test_data_path):
	train_X = np.load(train_data_path)
	test_X = np.load(test_data_path)
	train_Y = np.zeros((train_X.shape[0]),dtype='int32')
	for i in range(12500):
		train_Y[i] = 1
	test_Y = copy.deepcopy(train_Y)
	#shuffle the data 
	s = np.arange(train_X.shape[0])
	np.random.shuffle(s)
	train_X = train_X[s]
	train_Y = train_Y[s]
	test_X = test_X[s]
	test_Y = test_Y[s]
	return train_X,train_Y,test_X,test_Y
	
if __name__ == '__main__':
	train_X,train_Y,test_X,test_Y = get_data('train_id_mat.npy','test_id_mat.npy')

