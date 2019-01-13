# Obtain the run-time of convolution operations based on the amount of sparsity
# Using the blocksparse module developen by openai
# Date : 11-1-2019

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from blocksparse.conv import BlocksparseConv, BlocksparseDeconv
from blocksparse.norms import batch_norm, batch_norm_inference, batch_norm_inf_test, batch_norm_test, batch_norm_grad_test
import tensorflow as tf
import numpy as np
from random import shuffle
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import ops

#Define the input tensor with the given sparsity
#conv_tensor = BlocksparseConv()

B = 4
blockC = 32
blockK = 48
BCK_diagonal =  [
                    [
                        [b*blockC + c for c in range(blockC)],
                        [b*blockK + k for k in range(blockK)],
                    ] for b in range(B)
                ]

print("BCK_diagonal :" + str(BCK_diagonal))

x_shape = [100,64,64,3] # "NHWC format"
y_shape = [100,64,64,8] # "NHWC format"
w_shape = [3,3,3,8] # "FFCC_O" format

# Define a set of slice objects

sdim = slice(1,-1)
print("sdim :" + str(sdim))
fdim = slice(0,-2)
print("fdim :" + str(fdim))
cdim = -1

C = x_shape[cdim] # The number of channels in the input tensor
K = y_shape[cdim] # The number of kernels
print("Number of channels in the input tensor :" + str(C))
print("Number of kernels :" + str(K))

MPQ = np.expand_dims(y_shape[sdim],axis=0)
DHW = np.expand_dims(x_shape[sdim],axis=0)
TRS = np.expand_dims(w_shape[fdim],axis=0)

print("MPQ : "+str(MPQ))
print("DHW : "+str(DHW))
print("TRS : "+str(TRS))
