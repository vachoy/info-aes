#alternate gen_image following example code

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import _pickle as cPickle
import gzip

#import trained weights,masks
weights_filename = '4_8made_weightsv10.npz'
masks_filename = '4_8made_masks.pkl'
weights = np.load(weights_filename)
masks = cPickle.load(open(masks_filename, "rb"))

#activation functions
def sigmoid_np(x):
    z = 1/(1 + np.exp(-x))
    return z

def relu_np(x):
    z = np.maximum(x,0)
    return z

def display_images(x, digit_size=28,n=10):
	figure = np.zeros((digit_size*n,digit_size*n))

	for i in range(n):
		for j in range(n):
			index = np.random.randint(0, x.shape[0])
			digit = x[index].reshape(digit_size,digit_size)

			x=i*digit_size
			y=j*digit_size
			figure[x:x+digit_size,y:y+digit_size] = digit

	plt.figure(figsize=(n,n))
	plt.imshow(figure,cmap='gray')
	plt.show()

def gen_samples(weights,h1_mask, h2_mask, out_m, dir_m,num_samples=10,in_indexes):
	x_sample = np.random.rand(num_samples,28*28)

	for i in range(0,(28*28)):
		hidden1 = relu_np(np.add(weights['b1'],np.matmul(x,np.multiply(weights['w1'],h1_mask))))
        hidden2 = relu_np(np.add(weights['b2'],np.matmul(hidden1,np.multiply(weights['w2'],h2_mask))))
        x_out = sigmoid_np(np.add(np.add(weights['x_b_hat'],np.matmul(hidden2,np.multiply(weights['x_hat'],out_m))), 
        	np.matmul(x,np.multiply(weights['dirr'],dir_m))))

        p = np.random.rand(num_samples)
        index = in_indexes[i]
        x_sample[:,index] = (x_out[:,index] > p.astype(float))

    return x_sample

outputs = gen_samples(weights,masks['0'][0],masks['0'][1],masks['0'][2],masks['0'][3],masks['0'][4])
display_images(outputs,n=7)