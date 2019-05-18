#! usr/bin/env python3

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import _pickle as cPickle
import gzip
np.set_printoptions(threshold=np.nan)
import time

#load with cPickle
mnist = gzip.open('mnist.pkl.gz','rb')
train, val, test = cPickle.load(mnist, encoding='latin1')
mnist.close()

#define parameters
batch_size = 256
features = 784
hidden_layers = 2
hidden_units = 4000
steps = 500000
check = steps/10

#activation functions
def sigmoid_np(x):
    z = 1/(1 + np.exp(-x))
    return z

def relu_np(x):
    z = np.maximum(x,0)
    return z

#extract train set images and labels
images = train[0]
labels = train[1]

#binarize data
b_images = images

for i in range(len(images)):
    for j in range(784):
        if images[i][j] > 0:
            b_images[i][j] = 1

#generate data
def gen_data(batch_size):
  random_rows = np.random.choice(b_images.shape[0],size=batch_size,replace=False)
  x_data = b_images[random_rows,:]
  return x_data
"""

def gen_image(num_images, h1_mask, h2_mask, out_m, dir_m):
    x = tf.random.uniform((num_images, features))
    #iterate time
    for i in range(0,features):
        hidden1 = tf.nn.relu(tf.add(sess.run(b1),tf.matmul(x,tf.multiply(sess.run(w1),h1_mask))))
        hidden2 = tf.nn.relu(tf.add(sess.run(b2),tf.matmul(hidden1,tf.multiply(sess.run(w2),h2_mask))))
        out = tf.nn.sigmoid(tf.add(tf.add(sess.run(x_b_hat),tf.matmul(hidden2,tf.multiply(sess.run(x_hat),sess.run(out_m)))), tf.matmul(x,tf.multiply(sess.run(dirr),sess.run(dir_m)))))
    #set p to current pixel probabilities
    p = out[:,i]
    #set x to sample from Bernoulli distribution using parameter p
    x[:,i] = tf.convert_to_tensor(np.random.binomial(1,p,size=x[:,i].shape), dtype=tf.float32)

    return x
"""
random_init = True

"""
node_index: assigns indexes to layer nodes
input parameters: nodes, an int representing the number of nodes in the layer
                    that require index assignment
                  features, an int representing the number of features in the
                    input data
                  h, an int representing the layer number that needs indexes
                    assigned
                  prev_nodes, a list/array that represents all of the indices
                    represented in the previous layer. default is 1 because when
                  not  using np.amin and creating indices for the first hidden layer,
                    it ne 203, in <meeds to be 1. otherwise, replace with a list/array of
                    previous layer indices.
output: indexes, an array representing the assigned indices for the layer the
function was called for.
"""
def node_index(nodes, features, h, prev_nodes=[1]):
  if h == 0 or h == hidden_layers+1:
    indexes = np.arange(1,features+1)
    np.random.shuffle(indexes)
  else:
    indexes = np.random.randint(np.amin(prev_nodes),high=((features+1)-h),size=nodes)
  return indexes

"""
in_mask: generate a mask matrix for input or hidden layers not connected to the
  output layer
input parameters: prev_nodes, an array containing the indexes of the nodes of the
                  b_images  preve 203, in <mious layer
                  prev_h, an int representing the layer number of the previous
                    layer
                  indexes, an array containing the indexes of the nodes of the
                    layer for which the mask is being generated
output: mask, an array of 1's and 0's representing the desired layer mask
"""
def in_mask(prev_nodes,prev_h,indexes):
  indexes=indexes
  mask = np.zeros((len(prev_nodes),len(indexes)))
  for i in range(len(prev_nodes+1)):
    for j in range(len(indexes+1)):
      if indexes[j] >= prev_nodes[i]:
        mask[i,j] = 1
  return mask
"""
out_mask: generate a mask matrix for the hidden layer connected to the output
   layer x_data
input parameters: prev_nodes, a array containing the indexes of the nodes of the
                    previous layer
                  prev, an int representing the layer number of the previous
                    layer
                  indexes, an array containing the indexes of the nodes of the
                    layer for which the mask is being generated
output: mask, an array of 1's and 0's representing the output layer mask
"""
def out_mask(prev_nodes,prev_h,indexes):
  indexes = indexes
  mask = np.zeros((len(prev_nodes),len(indexes)))
  for i in range(len(prev_nodes+1)):
    for j in range(len(indexes+1)):
      if indexes[j] > prev_nodes[i]:
        mask[i,j] = 1
  return mask

masks = dict()
for i in range(1):
    in_indexes = node_index(features,features,0)
    h1_indexes = node_index(hidden_units,features,0+1,prev_nodes=in_indexes)
    h2_indexes = node_index(hidden_units,features,1+1,prev_nodes=h1_indexes)
    out_indexes = in_indexes
    h1_mask = in_mask(in_indexes,0,h1_indexes)
    h2_mask = in_mask(h1_indexes,1,h2_indexes)
    out_m = out_mask(h2_indexes,2,out_indexes)
    dir_m = out_mask(in_indexes, 0, out_indexes)
    masks[str(i)] = [h1_mask, h2_mask, out_m, dir_m, in_indexes]

# save masks
#np.savez("4_1made_masks", m = masks)
f = open("4_29masktestmask.pkl", "wb")
cPickle.dump(masks,f)
f.close()

#instantiate variables
tf.reset_default_graph()

x = tf.placeholder(tf.float32,shape=(batch_size,features)) #images

if random_init:
  #h1 weight and bias
  w1 = tf.get_variable("w1",shape=(features,hidden_units),initializer=tf.random_normal_initializer(0,0.5))
  b1 = tf.get_variable("b1",shape=(1,hidden_units),initializer=tf.random_normal_initializer(0,0.5))

  #h2 weight and bias
  w2 = tf.get_variable("w2",shape=(hidden_units,hidden_units),initializer=tf.random_normal_initializer(0,0.5))
  b2 = tf.get_variable("b2",shape=(1,hidden_units),initializer=tf.random_normal_initializer(0,0.5))

  #output layer weight and bias
  x_hat = tf.get_variable("x_hat",shape=(hidden_units,features),initializer=tf.random_normal_initializer(0,0.0000005))
  x_b_hat = tf.get_variable("x_b_hat",shape=(1,features),initializer=tf.random_normal_initializer(0,0.0000005))

  #direct connection
  dirr = tf.get_variable("dirr",shape=(features,features),initializer=tf.random_normal_initializer(0,0.0000005))
else:
  #h1 weight and bias
  w1 = tf.get_variable("w1",shape=(features,hidden_units),initializer=tf.constant_initializer(weights['w1']))
  b1 = tf.get_variable("b1",shape=(1,hidden_units),initializer=tf.constant_initializer(weights['b1']))

  #h2 weight and bias
  w2 = tf.get_variable("w2",shape=(hidden_units,hidden_units),initializer=tf.constant_initializer(weights['w2']))
  b2 = tf.get_variable("b2",shape=(1,hidden_units),initializer=tf.constant_initializer(weights['b2']))

  #output layer weight and bias
  x_hat = tf.get_variable("x_hat",shape=(hidden_units,features),initializer=tf.constant_initializer(weights['x_hat']))
  x_b_hat = tf.get_variable("x_b_hat",shape=(1,features),initializer=tf.constant_initializer(weights['x_b_hat']))

  #direct connection
  dirr = tf.get_variable("dirr",shape=(features,features),initializer=tf.constant_initializer(weights['dirr']))

#create network
hidden1 = tf.nn.relu(tf.add(b1,tf.matmul(x,tf.multiply(w1,h1_mask))))
hidden2 = tf.nn.relu(tf.add(b2,tf.matmul(hidden1,tf.multiply(w2,h2_mask))))
out = tf.nn.sigmoid(tf.add(tf.add(x_b_hat,tf.matmul(hidden2,tf.multiply(x_hat,out_m))), tf.matmul(x,tf.multiply(dirr,dir_m))))

def cross_entropy(x, y, axis=-1):
  safe_y = tf.where(tf.equal(x, 0.), tf.ones_like(y), y)
  return -tf.reduce_sum(x * tf.log(safe_y), axis)

def entropy(x,axis=-1):
  return cross_entropy(x,x,axis)
####loss = tf.reduce_sum(-x*tf.log(out)-(1-x)*tf.log(1-out))
loss=entropy(out)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
init = tf.global_variables_initializer()

#run Session      # save current weights
with tf.Session() as sess:
  sess.run(init)
  counter = 0
  losses = []
  # randomly select input images to be used for auto-encoding capacity visualization
  """
  samples = 2
  input_x = np.empty((28*samples,28*samples))
  test_data = gen_data(batch_size)
  for j in range(samples):
            test_data = test_data[:,:784]
            input_x[i*28:(i+1)*28,j*28:(j+1)*28] = test_data[j].reshape([28,28])
  plt.imshow(input_x, origin="upper", cmap="gray")
  plt.title("Test Input Images")
  plt.savefig("4_29test_images.png")
  """
  for i in range(steps):
    count = 0 # this is different from counter
    h1_mask, h2_mask, out_m, dir_m = masks[str(count)][0], masks[str(count)][1],masks[str(count)][2], masks[str(count)][3]
    #if count == 9:
    #    count = 0
    #count+= 1
    x_data = gen_data(batch_size)
    _ = sess.run(optimizer,feed_dict={x:x_data})
    if np.mod(i,check) == 0:
      counter += 1
      losses.append(sess.run(loss, feed_dict={x:x_data}))
      #save current weights
      np.savez("4_29masktestv{}".format(counter), w1 = sess.run(w1),
                     b1 = sess.run(b1),
                     w2 = sess.run(w2),
                     b2 = sess.run(b2),
                     x_hat = sess.run(x_hat),
                     dirr = sess.run(dirr),
                     x_b_hat = sess.run(x_b_hat),
                     losses = losses)
      samples = 1
      input_x = np.empty((28*samples,28*samples))
      output_x = np.empty((28*samples,28*samples))

      for ii in range(samples):
        xx = gen_data(batch_size)
        test_data = sess.run(out, feed_dict={x:xx})

        for j in range(samples):
          xx = xx[:,:784]
          input_x[ii*28:(ii+1)*28,j*28:(j+1)*28] = xx[j].reshape([28,28])

        for j in range(samples):
          test_data = test_data[:,:784]
          output_x[ii*28:(ii+1)*28,j*28:(j+1)*28] = test_data[j].reshape([28,28])

      plt.imshow(input_x,origin="upper",cmap="gray")
      plt.title("Test Input Images")
      plt.savefig("4_29test_imagesinputv{}.png".format(counter))

      plt.imshow(output_x,origin="upper",cmap="gray")
      plt.title("Test Output Images")
      plt.savefig("4_29test_outputv{}.png".format(counter))
      # visualize current auto-encoding capacity
