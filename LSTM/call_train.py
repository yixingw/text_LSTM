import os

# list_of_filters = [16,32,64,128,256,512]
# 
# for filter_size in list_of_filters:
# 	command = "python train_CNN.py -ckpt_dir checkpoint"+str(filter_size)+'/ -num_filters '+str(filter_size)+' -save_file_name '+'error_hist_'+str(filter_size)+'.p'
# 	os.system(command)

#test the performance of different drop out

# list_drop_out = [0.1,0.3,0.5,0.7,0.9]
# 
# for drop_out in list_drop_out:
# 	command = "python train_LSTM.py -ckpt_dir checkpoint_drop_out_"+str(drop_out)+'/ -drop_out '+str(drop_out)+' -save_file_name '+'error_hist_drop_out'+str(int(drop_out*10))+'.p'
# 	print command
# 
# 	os.system(command)

#test the performance for different learning rate

# learning_rate_list = [0.5,0.1,0.01,0.01]
# for learing_rate in learning_rate_list:
# 	command = "python train_LSTM.py -ckpt_dir checkpoint_learing_rate"+str(learing_rate)+'/ -learning_rate '+str(learing_rate)+' -save_file_name '+'error_learning_rate'+str(learing_rate)+'.p'
# 	print command
# 
# 	os.system(command)

# hidden_sizes = [16,32,64,128,256]
# for hidden_size in hidden_sizes:
# 	command = "python train_LSTM.py -ckpt_dir checkpoint_hidden_size_"+str(hidden_size)+'/ -hidden_size '+str(hidden_size)+' -save_file_name '+'error_hist_hidden_size'+str(hidden_size)+'.p'
# 	print command
# 
# 	os.system(command)


# Test the performance of different optimization tech 
opt_methods = ['SGD','Momentum']
for opt in opt_methods:
	command = "python train_LSTM.py -ckpt_dir checkpoint_opt_method_"+opt+'/ -opt_method '+opt+' -save_file_name '+'error_hist_opt_method_'+opt+'.p'
	print command
	os.system(command)