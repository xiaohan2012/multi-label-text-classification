import pytest
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from eval_helpers import tf_precision_at_k, label_lists_to_sparse_tuple, \
    precision_at_ks


@pytest.fixture
def pred_value_1():
    # [0, 1], [2, 1], [0, 2]
    return np.array([[0.9, 0.8, 0.7], [0.7, 0.8, 0.9], [0.9, 0.7, 0.8]], dtype=np.float32)


@pytest.fixture
def pred_value_2():
    # [2, 1], [0, 1], [2, 1]
    return np.array([[0.7, 0.8, 0.9], [0.9, 0.8, 0.7], [0.7, 0.8, 0.9]], dtype=np.float32)


@pytest.fixture
def correct_values():
    return [[0, 1, 2], [1, 2], [0, 2]]

    
def test_tf_precision_at_k(pred_value_1, pred_value_2, correct_values):
    n_classes = 3

    sparse_tensor_tuple = label_lists_to_sparse_tuple(correct_values, n_classes)

    with tf.Session() as sess:
        print(tf.sparse_to_dense(sparse_tensor_tuple[0],
                                 sparse_tensor_tuple[2],
                                 sparse_tensor_tuple[1]).eval())

        pred = tf.placeholder(tf.float32, shape=[None, None], name='pred')
        correct_labels = tf.sparse_placeholder(tf.int32, shape=[None, n_classes], name='correct_labels')
        precision = tf_precision_at_k(pred, correct_labels, k=2)
        
        p1 = sess.run(precision,
                      feed_dict={
                          pred: pred_value_1,
                          correct_labels: sparse_tensor_tuple
                      })
        
        assert np.isclose(p1, 1.0)
        
        p2 = sess.run(precision,
                      feed_dict={
                          pred: pred_value_2,
                          correct_labels: sparse_tensor_tuple
                      })
        assert np.isclose(p2, np.mean([1, 0.5, 0.5]))


def test_precision_at_k_dense(pred_value_1, pred_value_2, correct_values):
    p1 = precision_at_ks(pred_value_1, correct_values, ks=[2])[0]
    assert np.isclose(p1, 1.0)
        
    p2 = precision_at_ks(pred_value_2, correct_values, ks=[2])[0]
    assert np.isclose(p2, np.mean([1, 0.5, 0.5]))
        

def test_precision_at_k_sparse(pred_value_1, pred_value_2, correct_values):
    p1 = precision_at_ks(csr_matrix(pred_value_1), correct_values, ks=[2])[0]
    assert np.isclose(p1, 1.0)
        
    p2 = precision_at_ks(csr_matrix(pred_value_2), correct_values, ks=[2])[0]
    assert np.isclose(p2, np.mean([1, 0.5, 0.5]))
