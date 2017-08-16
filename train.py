import os
import numpy as np
import pandas as pd
import time
import pickle as pkl
import tensorflow as tf

from tensorflow.contrib import learn
from sklearn.preprocessing import MultiLabelBinarizer

from data_helpers import batch_iter
from text_cnn import TextCNN


tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# model parameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


data_dir = 'data/stackexchange/datascience/'

input_text = pd.read_csv(os.path.join(data_dir, 'input_text.csv'), header=None)[1].tolist()
labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'), header=None)[0].tolist()
labels = list(map(lambda s: s.split(','), labels))


# In[34]:


# filter out too long documents
max_doc_len = 1000
tpls = [(t, l) for t, l in zip(input_text, labels) if len(t.split()) <= max_doc_len]
input_text, labels = list(zip(*tpls))

text_processor = learn.preprocessing.VocabularyProcessor(max_doc_len)
x = np.array(list(text_processor.fit_transform(input_text)))


mb = MultiLabelBinarizer()
y = mb.fit_transform(labels)

# split it
np.random.seed(12345)

ind = np.random.permutation(x.shape[0])
shuffled_x = x[ind, :]
shuffled_y = y[ind, :]

idx = int(FLAGS.dev_sample_percentage * x.shape[0])
train_x, train_y = shuffled_x[idx:, :], shuffled_y[idx:, :]
dev_x, dev_y = shuffled_x[:idx, :], shuffled_y[:idx, :]


with tf.Session() as sess:
    
    cnn = TextCNN(max_doc_len, len(mb.classes_), FLAGS.embedding_dim, len(text_processor.vocabulary_),
                  list(map(int, FLAGS.filter_sizes.split(','))),
                  FLAGS.num_filters)
    # train operation
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-5)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # IO direction stuff
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

    
    # summary writer
    train_summary_dir = os.path.join(out_dir, "summary/train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_dir = os.path.join(out_dir, "summary/dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # checkpoint writer
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # summary operation
    grad_summaries = []
    for grad, v in grads_and_vars:
        if grad is not None:
            hist = tf.summary.histogram("{}/grad/hist".format(v.name), grad)
            sparsity = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(grad))
            grad_summaries.append(hist)
            grad_summaries.append(sparsity)
    grad_summary = tf.summary.merge(grad_summaries)

    prec_summary = tf.summary.scalar("precision", cnn.precision)
    rec_summary = tf.summary.scalar("recall", cnn.recall)
    loss_summary = tf.summary.scalar("loss", cnn.loss)

    train_summary_op = tf.summary.merge([grad_summary, prec_summary, rec_summary, loss_summary])
    dev_summary_op = tf.summary.merge([prec_summary, rec_summary, loss_summary])
    
    # real code
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # dump vectorizer
    text_processor.save(os.path.join(out_dir, 'text_processor'))
    pkl.dump(mb, open(os.path.join(out_dir, 'label_encoder.pkl'), 'wb'))
    
    data = list(zip(train_x, train_y))
    batches = batch_iter(data, FLAGS.batch_size, FLAGS.num_epochs)
    for batch in batches:
        batch_x, batch_y = zip(*batch)
        feed_dict = {cnn.input_x: batch_x,
                     cnn.input_y: batch_y,
                     cnn.dropout_keep_prob: FLAGS.dropout_keep_prob}
        
        current_step = tf.train.global_step(sess, global_step)
        _, current_step, summaries, loss, prec, rec = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.precision, cnn.recall],
            feed_dict=feed_dict)
        print("(TRAIN) at step {}: loss={:.2f}, precision={:4f}, recall={:4f}".format(current_step, loss, prec, rec))
        train_summary_writer.add_summary(summaries, current_step)

        if current_step % FLAGS.evaluate_every == 0:
            loss, summaries, prec, rec = sess.run(
                [cnn.loss, dev_summary_op, cnn.precision, cnn.recall],
                feed_dict={cnn.input_x: dev_x, cnn.input_y: dev_y, cnn.dropout_keep_prob: 1})
            print("(DEV) at step {}: loss={:2f}, precision={:4f}, recall={:4f}".format(
                current_step, loss, prec, rec))
            dev_summary_writer.add_summary(summaries, current_step)
            
        if current_step % FLAGS.checkpoint_every == 0:
            saver.save(sess, checkpoint_prefix, global_step=global_step)

