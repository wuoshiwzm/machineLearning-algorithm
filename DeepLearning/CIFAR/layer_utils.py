"""
神经网络工具函数类
"""
from layers import *
from fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b) #affine_forward 就是计算 w*x+b
  # 因为输入变换成输出后的值，再下一次反向传播时时还要用，所有存在变量relu_cache里
  out, relu_cache = relu_forward(a) #relu层，把数据做一个非线性的映射
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass

#############################################################################
##  help layer  affine_bn_relu   start
#############################################################################
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  affine - batchnorm - ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta, bn_param parameters for batchnorm layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a_out, fc_cache = affine_forward(x, w, b)
  b_out,  bn_cache = batchnorm_forward(a_out, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b_out)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache


def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine-bn-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  db = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward_alt(db, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta
#############################################################################
##  help layer   affine_bn_relu    end
#############################################################################

#############################################################################
##  help layer    affine_bn_relu_dp   start
#############################################################################
def affine_bn_relu_dp_forward(x, w, b, gamma, beta, bn_param, dropout_param):
  """
  affine - batchnorm - ReLU - dropout

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta, bn_param parameters for batchnorm layer
  - dropout_param parameters for dropout layer

  Returns a tuple of:
  - out: Output from the dropout
  - cache: Object to give to the backward pass
  """
  a_out, fc_cache = affine_forward(x, w, b)
  b_out,  bn_cache = batchnorm_forward(a_out, gamma, beta, bn_param)
  r_out, relu_cache = relu_forward(b_out)
  out, dp_cache = dropout_forward(r_out, dropout_param)
  cache = (fc_cache, bn_cache, relu_cache, dp_cache)
  return out, cache


def affine_bn_relu_dp_backward(dout, cache):
  """
  Backward pass for the affine-bn-relu-dp convenience layer
  """

  fc_cache, bn_cache, relu_cache, dp_cache = cache
  d_dp = dropout_backward(dout, dp_cache)
  db = relu_backward(d_dp, relu_cache)
  da, dgamma, dbeta = batchnorm_backward_alt(db, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta
#############################################################################
##  help layer    affine_bn_relu_dp    end
#############################################################################

def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


################### help layer ######################################
def conv_bn_relu_pool_forward(x, w, b, gamma, beta, \
                              conv_param, pool_param, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer
  - bn_param: Parameters for the batch normalization layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(bn)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache


def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, conv_cache)
  return dx, dw, db, dgamma, dbeta

