# https://github.com/aladdinpersson/Machine-Learning-Collection

import tensorflow as tf
import numpy as np
import os 
# make sure the enviroment config 
# set TF_ENABLE_ONEDNN_OPTS=0
os.environ['TFF_CPP_MIN_LOG_LEVEL']='2'

# working with GPU instead CPU - faster efficnt...
# print( tf.config.list_logical_devices('GPU'))
# pd=tf.config.list_logical_devices('GPU')
# tf.config.experimental.set_memory_growth(pd[0],True)

# oneDNN: oneDNN (Deep Neural Network Library) is optimized for performance on Intel CPUs. 
# When enabled, it can provide significant speedups
# for many deep learning operations by leveraging Intel's architecture-specific optimizations.


# create a number in numpy (simple int)
# x= tf.constant(3)
# print(x.numpy())

# create matrix
# y= np.ones([2,3], dtype=None, order='C')
# print(y)

# create matrix with normalize numbers
# x=tf.random.normal((2,2),mean=0,stddev=2)
# print(x)
# print(x.numpy()[0][0]) #get the first number 

# create matrix with min, max
# x=tf.random.uniform((2,2),minval=2,maxval=4)
# print(x)
# print(x.numpy()[0][0]) #get the first number 


# create range vector
# x=tf.range(5)
# print(x)
# print(x.numpy()) #get the first number 

# create range with felta
# x=tf.range(start=2,limit=20,delta=3)
# print(x)

# casting
# print(tf.cast(x,dtype=tf.int16))

# connect vectors
# x=tf.constant([4,5,6])
# y=tf.constant([7,8,9])

# print(tf.add(x,y))
# print(x+y)
# print(tf.subtract(x,y))
# print(x-y)


x=tf.constant([4,5,6,7,8,9,3])
y=tf.constant([7,8,9])

# z=tf.tensordot(x,y,axes=1)
# print(z)

z= x# **3
# print(z[:])
# print(z[1:]) # skip first
# print(z[1:3]) # 1-3 range
# print(z[::2]) # every other
# print(z[::-1]) # revers
# print(tf.gather(x,[2,4])) # get items 2,4..

# matrix
# x=tf.constant([[2,3],[4,5],[7,8]])
# print(x[0,:])
# print(x[0:2,:])

x=tf.range(start=0,limit=12,delta=1)
# print(x)
# print(tf.reshape(x,(3,4)))
tf.experimental.numpy.experimental_enable_numpy_behavior()
x=tf.reshape(x,(3,4))
print(x.T)
print(tf.transpose(x))
