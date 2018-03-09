# -*- coding: utf-8 -*-
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from LSTM_model import TextRNN
import os
import pickle
from get_train_test_data import get_data
from create_voc_index import load_pretraind_embedding

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",2,"number of label")
tf.app.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.") #批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","text_rnn_checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",200,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",50,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",60,"epoch number")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("save_file_name",'train_hist.p','output hsitory file name')
tf.app.flags.DEFINE_float("drop_out",0.5,"drop out ratio")
tf.app.flags.DEFINE_integer("hidden_size",50,"number of LSTM hidden unit")
vocab_size  = 400000 #no use
#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #1. Get the train and test data 
    trainX,trainY,testX,testY = get_data('../../datafile/train_id_mat.npy','../../datafile/test_id_mat.npy')
    train_loss_hist,train_acc_hist,test_loss_hist,test_acc_hist = [],[],[],[]
    print "The size of the training data is ", trainX.shape[0]
    print "The size of the testing data is ", testX.shape[0]
    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        textRNN=TextRNN(FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sequence_length,
        vocab_size, FLAGS.embed_size, FLAGS.is_training,FLAGS.hidden_size)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                _,word_Embedding = load_pretraind_embedding("../../datafile/glove.6B.50d.txt")
                t_assign_embedding = tf.assign(textRNN.Embedding,word_Embedding)  # assign this value to our embedding variables of our model.
                sess.run(t_assign_embedding)
        curr_epoch=sess.run(textRNN.epoch_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])#;print("trainY[start:end]:",trainY[start:end])
                curr_loss,curr_acc,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.train_op],feed_dict={textRNN.input_x:trainX[start:end],textRNN.input_y:trainY[start:end]
                    ,textRNN.dropout_keep_prob:FLAGS.drop_out}) #curr_acc--->TextCNN.accuracy -->,textRNN.dropout_keep_prob:1
                loss,counter,acc=loss+curr_loss,counter+1,acc+curr_acc
                if counter %1==0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
                    train_loss_hist.append(loss/float(counter)) 
                    train_acc_hist.append(acc/float(counter))
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(textRNN.epoch_increment)
            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                # eval_loss, eval_acc=do_eval(sess,textRNN,testX,testY,batch_size,vocabulary_index2word_label)
                eval_loss, eval_acc=do_eval(sess,textRNN,testX,testY,batch_size)
                test_loss_hist.append(eval_loss)
                test_acc_hist.append(eval_acc)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch,eval_loss,eval_acc))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)
        test_loss, test_acc = do_eval(sess, textRNN, testX, testY, batch_size)
        train_hist = [train_loss_hist,train_acc_hist,test_loss_hist,test_acc_hist]
        pickle.dump(train_hist,open(save_file_name,'wb'))
    pass


def do_eval(sess,textRNN,evalX,evalY,batch_size):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        curr_eval_loss, logits,curr_eval_acc= sess.run([textRNN.loss_val,textRNN.logits,textRNN.accuracy],#curr_eval_acc--->textCNN.accuracy
                                          feed_dict={textRNN.input_x: evalX[start:end],textRNN.input_y: evalY[start:end]
                                              ,textRNN.dropout_keep_prob:1})
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)

#从logits中取出前五 get label using logits
def get_label_using_logits(logits,top_number=1):
    #print("get_label_using_logits.logits:",logits) #1-d array: array([-5.69036102, -8.54903221, -5.63954401, ..., -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    #label_list=[]
    #for index in index_list:
    #    label=vocabulary_index2word_label[index]
    #    label_list.append(label) #('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return index_list

def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

if __name__ == "__main__":
    tf.app.run()