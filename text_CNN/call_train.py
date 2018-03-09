import os

list_of_filters = [16,32,64,128,256,512]

for filter_size in list_of_filters:
	command = "python train_CNN.py -ckpt_dir checkpoint"+str(filter_size)+'/ -num_filters '+str(filter_size)+' -save_file_name '+'error_hist_'+str(filter_size)+'.p'
	print command
