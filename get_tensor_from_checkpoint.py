
# coding: utf-8

import tensorflow as tf


checkpoint_file = 'runs/deepwalk/checkpoints/model-25000'

sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
saver.restore(sess, checkpoint_file)

embedding_table = sess.graph.get_operation_by_name('embedding/table')

val = embedding_table.outputs[0].eval()

print(val)

