import pickle

def make_plot(hist_file):
	trian_hist = pickle.load(open(hist_file,'rb'))
	[train_loss_hist,train_acc_hist,test_loss_hist,test_acc_hist] = trian_hist
	print train_loss_hist
	print train_acc_hist
	print test_loss_hist
	print test_acc_hist
	pass
if __name__ == '__main__':
	make_plot('train_hist.p')