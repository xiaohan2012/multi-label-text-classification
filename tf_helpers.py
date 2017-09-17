import tensorflow as tf


def flatten(t):
    """flatten tensor of any dimension to 1d"""
    return tf.reshape(t, [tf.reduce_prod(t.shape)])


def dynamic_max_k_pool(t, k):
    """
    perform dynamic max-k pooling on t,

    note that:

    1. the 
    2. only supports 1d data for now

    Param:
    -----------
    t: Tensor, 2d (batches, repr)
    k: int

    Return:
    -----------
    Tensor, 2d (batches, k)
    """
