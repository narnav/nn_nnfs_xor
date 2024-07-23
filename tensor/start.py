import tensorflow as tf
import numpy as np

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
x=tf.range(5)
print(x)
print(x.numpy()) #get the first number 



# x=tf.ones((2,3))
# print(x.numpy())
# print(y)
# print (x)

# set TF_ENABLE_ONEDNN_OPTS=0