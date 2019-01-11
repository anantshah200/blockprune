# Obtain the run-time of convolution operations based on the amount of sparsity
# Using the blocksparse module developen by openai
# Date : 11-1-2019

from blocksparse.conv import BlocksparseConv, BlocksparseDeconv
from blocksparse.norms import batch_norm, batch_norm_inference, batch_norm_inf_test, batch_norm_test, batch_norm_grad_test
import tensorflow as tf
import numpy as np
from random import shuffle
from tensorflow.python.ops import gradient_checker

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
