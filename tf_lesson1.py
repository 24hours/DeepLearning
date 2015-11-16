from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from algo.helper import loaddata_tf
from sklearn import preprocessing
import numpy as np


lb = preprocessing.LabelBinarizer()
mnist = loaddata_tf("./data/mnist.pkl.gz")
train_x, train_y = mnist[0]
valid_x, valid_y = mnist[1]
test_x, test_y = mnist[2]

batch_no = 2000
batch_size = train_x.shape[0] / batch_no

lb.fit(range(10))
train_y = lb.transform(train_y)
valid_y = lb.transform(valid_y)
test_y = lb.transform(test_y)

# Create the modela

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
# Define loss and optimizer
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
# Train

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Training starting")
done_train = False
valid_score = 0
while done_train is False:
	for i in range(batch_no):
		ind = i * batch_size
		batch_xs = train_x[ind: ind+batch_no]
		batch_ys = train_y[ind: ind+batch_no]
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	print("train accuracy", sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
	v_s = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
	print("   valid accuracy", v_s)
	
	improvement =  v_s - valid_score
	valid_score = v_s

	if(improvement < 0.001):
		done_train = True
# Test trained model

print("train accuracy", sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
print("test accuracy", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
