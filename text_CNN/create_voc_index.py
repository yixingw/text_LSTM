import tensorflow as tf
import numpy as np
import re
from os import listdir
from os.path import isfile, join


#load pretrained word vectors

def load_pretraind_embedding(glove_file):
	print "loading the glove embedding"
	word_list = []
	embedding_list = []
	with open(glove_file,'r') as f:
		for line in f:
			line_split = line.split()
			word = line_split[0]
			word_list.append(word)
			embedding = np.asarray(line_split[1:],dtype = "float32")
			embedding_list.append(embedding)
	return word_list,np.asarray(embedding_list)

def pre_process_text(string):
	strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
	string = string.lower().replace("<br />", " ")
	return re.sub(strip_special_chars, "", string.lower())

def get_embedding_dic(path_to_pos,path_to_neg,outputfile,wordsList):
	positiveFiles = [path_to_pos + f for f in listdir(path_to_pos) if isfile(join(path_to_pos, f))]
	negativeFiles = [path_to_neg + f for f in listdir(path_to_neg) if isfile(join(path_to_neg, f))]
	maxSeqLength = 200
	ids = np.zeros((25000, maxSeqLength), dtype='int32')
	fileCounter = 0
	for pf in positiveFiles:
	   with open(pf, "r") as f:
	       indexCounter = 0
	       line=f.readline()
	       cleanedLine = pre_process_text(line)
	       split = cleanedLine.split()
	       for word in split:
	           try:
	               ids[fileCounter][indexCounter] = wordsList.index(word)
	           except ValueError:
	               ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
	           indexCounter = indexCounter + 1
	           if indexCounter >= maxSeqLength:
	               break
	       fileCounter = fileCounter + 1 

	for nf in negativeFiles:
	   with open(nf, "r") as f:
	       indexCounter = 0
	       line=f.readline()
	       cleanedLine = pre_process_text(line)
	       split = cleanedLine.split()
	       for word in split:
	           try:
	               ids[fileCounter][indexCounter] = wordsList.index(word)
	           except ValueError:
	               ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
	           indexCounter = indexCounter + 1
	           if indexCounter >= maxSeqLength:
	               break
	       fileCounter = fileCounter + 1 
	#Pass into embedding function and see if it evaluates. 
	np.save(outputfile, ids)

def load_text_test(path_to_pos,path_to_neg):

	pass


if __name__ == '__main__':
	word_list,embedding_vec = load_pretraind_embedding("./glove.6B.50d.txt")
	get_embedding_dic('aclImdb/test/pos/','aclImdb/test/neg/','test_id_mat',word_list)
	# word_id = np.load('./train_id_mat.npy')
	# print word_id

