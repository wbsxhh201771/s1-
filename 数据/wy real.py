import tensorflow as tf
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame,Series

plt.rcParams['font.sans-serif']=['FangSong']
data=pd.read_excel('网约车.xls')
data1=DataFrame(data,columns=['指标值'])
data2=data1['指标值']
x=range(2311)
HIDDEN_SIZE = 30                        
NUM_LAYERS = 2                           
TIMESTEPS = 10                           
TRAINING_STEPS = 1000000                  
BATCH_SIZE = 32                            
TRAINING_EXAMPLES = 2100                   
TESTING_EXAMPLES = 211                    
SAMPLE_GAP = 0.00001
'''
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean=tf.reduce_mean(var)
        tf.summary.scalar('mean/'+name, mean)
        stddev=tf.sqrt(tf.reduce_mean(tf.sqrt(var-mean)))
        tf.summary.scalar('stddev/'+name, stddev)
'''                         
def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  

def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) 
        for _ in range(NUM_LAYERS)])    

    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]

   
    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=None)
    
    if not is_training:
        return predictions, None, None
        
    loss = tf.compat.v1.losses.mean_squared_error(labels=y, predictions=predictions)

    
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.compat.v1.train.get_global_step(),
        optimizer="Adagrad", learning_rate=0.5)
    return predictions, loss, train_op
def run_eval(sess, test_X, test_y): 
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)
    
    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)
    
    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_wy')
    plt.legend()
    plt.savefig("wy.png",bbox_inches='tight')
    #plt.show()

def train(sess, test_X, train_y):
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)
    
    
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        
        if i % 1000 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))
        

test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.array(list(data2)[0:TESTING_EXAMPLES],dtype=np.float32))
test_X, test_y =  generate_data(np.array(list(data2)[TESTING_EXAMPLES:],dtype=np.float32))
with tf.Session() as  sess:
    train(sess,train_X,train_y)
    run_eval(sess, test_X,test_y)