import numpy as np
import tensorflow as tf

from eval_helpers import tf_precision_at_k, label_lists_to_sparse_tuple


def test_tf_precision_at_k():
    # [0, 1], [2, 1], [0, 2]
    pred_value_1 = np.array([[0.9, 0.8, 0.7], [0.7, 0.8, 0.9], [0.9, 0.7, 0.8]], dtype=np.float32)

    # [2, 1], [0, 1], [2, 1]
    pred_value_2 = np.array([[0.7, 0.8, 0.9], [0.9, 0.8, 0.7], [0.7, 0.8, 0.9]], dtype=np.float32)
    
    correct_values = [[0, 1, 2], [1, 2], [0, 2]]
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
