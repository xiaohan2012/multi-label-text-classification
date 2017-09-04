import pytest
import numpy as np
import tensorflow as tf

from eval_helpers import tf_precision_at_k


def test_tf_precision_at_k():
    pred_value_1 = np.array([[0.9, 0.8, 0.7], [0.7, 0.8, 0.9], [0.9, 0.7, 0.8]], dtype=np.float32)
    pred_value_2 = np.array([[0.7, 0.8, 0.9], [0.9, 0.8, 0.7], [0.7, 0.8, 0.9]], dtype=np.float32)
    correct_value = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
    
    with tf.Session() as sess:
        pred = tf.placeholder(tf.float32, shape=[None, None], name='pred')
        correct_labels = tf.placeholder(tf.int32, shape=[None, None], name='correct_labels')
        precision = tf_precision_at_k(pred, correct_labels, k=2)

        p1 = sess.run(precision,
                      feed_dict={
                          pred: pred_value_1,
                          correct_labels: correct_value
                      })
        
        assert np.isclose(p1, 1.0)
        
        p2 = sess.run(precision,
                      feed_dict={
                          pred: pred_value_2,
                          correct_labels: correct_value
                      })
        assert np.isclose(p2, 0.5)
