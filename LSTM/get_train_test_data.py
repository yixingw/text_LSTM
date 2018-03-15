import numpy as np 
import copy
from sklearn.model_selection import train_test_split

def get_data(train_data_path,test_data_path):
	train_X = np.load(train_data_path)
	test_X = np.load(test_data_path)
	train_Y = np.zeros((train_X.shape[0]),dtype='int32')
	for i in range(12500):
		train_Y[i] = 1
	test_Y = copy.deepcopy(train_Y)
	#concate the train and text data
	data_X = np.concatenate((train_X,test_X),axis = 0)
	data_Y = np.concatenate((train_Y,test_Y),axis = 0)

	#split the data 
	X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

	return X_train,y_train,X_test,y_test
	
if __name__ == '__main__':
	train_X,train_Y,test_X,test_Y = get_data('train_id_mat.npy','test_id_mat.npy')

