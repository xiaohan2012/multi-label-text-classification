import numpy as np
import tensorflow as tf


def precision(p, t):
    """>>> precision({1, 2, 3, 4}, {1})
       0.25
    """
    return len(t.intersection(p)) / len(p)


def precision_at_ks(Y_pred_scores, Y_test, ks=[1, 3, 5, 10]):
    k = 3
    for k in [1, 3, 5, 10]:
        Y_pred = []
        for i in np.arange(Y_pred_scores.shape[0]):
            idx = np.argsort(Y_pred_scores[i].data)[::-1]
            Y_pred.append(set(Y_pred_scores[i].indices[idx[:k]]))

        precision_at_k = np.mean([precision(set(yt), yp) for yt, yp in zip(Y_test, Y_pred)])
        print('precision at {}: {}'.format(k, precision_at_k))


def tf_precision_at_k(pred_values, correct_labels, k):
    _, pred_labels = tf.nn.top_k(pred_values, k=k, sorted=True)

    num_intersections = tf.sets.set_size(tf.sets.set_intersection(pred_labels, correct_labels))

    return tf.reduce_mean(tf.divide(num_intersections, k))
