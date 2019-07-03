# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 21:11:53 2019

@author: HJ
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:27:48 2019

@author: HJ
"""
'''
import librosa
import os,glob
'''
import tensorflow as tf
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(103)  # reproducibility

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

tf.reset_default_graph()
frame_length = 0.025
frame_stride = 0.0785
'''
train_path= 'C:\\Project\\Total Data\\' # test 세트 본인 컴퓨터 주소에 맞게 
#test_path = 'C:\\Project\\Eval\\' # eval 세트 본인 컴퓨테 주소에 맞게

train=[]
train_label_data = np.loadtxt("C:\\Project\\baby_crying_total_label.csv", delimiter=",",dtype=np.float32)
train_label=train_label_data[:,1:]




for filename in glob.glob(os.path.join(train_path, '*wav')):
    
            j, sr = librosa.load(filename, sr = 44100, offset = 0, duration = 6) # filename = 파일,sr = sampling_rate, offset = wav 몇초부터, duration = (offset + duration)초까지 (ex) 5.00~15.00까지 
            input_nfft = int(round(sr*frame_length)) # 음성 길이를 얼만큼으로 자를것인가?
            input_stride = int(round(sr*frame_stride))
            train_mel = librosa.feature.melspectrogram(j, n_mels = 128, n_fft = input_nfft, hop_length=input_stride)            
            train.append((train_mel)) # test 세트에 mel 담기
'''     

# hop_length = 음성의 magnitude를 얼만큼 겹친 상태로 잘라서 보여줄 것인가?
            
# hyper parameters

data=np.load('C:\\Project\\baby_crying_feature_mel.npz')
f_train=data['train']
f_train_label=data['train_label']

train_random = []

for i in range(len(f_train)):
    train_random_set = [f_train[i], f_train_label[i]]
    train_random.append(train_random_set)

random.shuffle(train_random)

train = []
train_label = []

for i in range(len(f_train)):
    train.append(train_random[i][0])
    train_label.append(train_random[i][1])

train = np.array(train)
train_label = np.array(train_label)

learning_rate = 0.001
training_epochs = 20
batch_size = 20

'''
for i in range(len(train)):
    if train[i].shape[1] != 72:
        train[i] = train[i][:,0:72]
'''


class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 128,72])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 128, 72, 1])
            self.Y = tf.placeholder(tf.float32, [None, 2])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[4, 4],
                                     padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1,
                                         rate=0.3, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[4, 4],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2,
                                         rate=0.3, training=self.training)

            # Convolutional Layer #3 and Pooling Layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[4, 4],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3,
                                         rate=0.3, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(dropout3, [-1, 128 * 16 * 9])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=576, activation=tf.nn.relu) 
            dropout4 = tf.layers.dropout(inputs=dense4,
                                         rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=dropout4, units=2)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()

models = []
num_models = 2
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())
print('Learning Started!')

# train my model
for epoch in range(training_epochs):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(len(train)/batch_size)
    for i in range(total_batch):
        batch_xs = train[i*batch_size:i*batch_size+batch_size]
        batch_ys = train_label[i*batch_size:i*batch_size+batch_size]

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
print('Learning Finished!')


# Test model and check accuracy
test_size = len(train_label)
predictions = np.zeros([test_size, 2])
for m_idx, m in enumerate(models):    
    p = m.predict(train)
    predictions += p
    print(m_idx, 'Accuracy:', m.get_accuracy(train, train_label))

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(train_label, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))
