from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from algo.helper import loaddata_tf
from sklearn import preprocessing
import numpy as np

# import tensorf.mnist as input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


lb = preprocessing.LabelBinarizer()

mnist = loaddata_tf("./data/mnist.pkl.gz")
train_x, train_y = mnist[0]
test_x, test_y = mnist[2]

batch_no = 1000
batch_size = train_x.shape[0] / batch_no

lb.fit(range(10))
train_y = lb.transform(train_y)
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

for i in range(1000):
	# original code use this, but our code load mnist differently
	# batch_xs, batch_ys = mnist.train.next_batch(100)
	ind = i * batch_size
	batch_xs = train_x[ind: ind+batch_no]
	batch_ys = train_y[ind: ind+batch_no]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))

print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))