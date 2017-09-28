# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import datetime

from tensorflow.contrib import learn

from kim_cnn import KimCNN
from eval_helpers import label_lists_to_sparse_tuple
from data_helpers import batch_iter, load_pickle


# In[4]:

tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')
tf.flags.DEFINE_integer('tag_freq_threshold', 5, 'minimum frequency of a tag')

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("max_document_length", 2000, "Maximum length of document, exceeding part is truncated")

# Architecutural parameters

tf.flags.DEFINE_string("loss_function", 'sigmoid', "loss function: (softmax|sigmoid) (Default: sigmoid)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 1)")  # our storage quota is low

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

data_dir = FLAGS.data_dir
dataset_id = list(filter(None, data_dir.split('/')))[-1]
print('dataset_id:', dataset_id)

# load data
# ===============================================
train_text, dev_text, _ = load_pickle(
    os.path.join(data_dir, "text_split.pkl"))
y_id_train, y_id_dev, _ = load_pickle(
    os.path.join(data_dir, "labels_id_split.pkl"))
y_binary_train, y_binary_dev, _ = load_pickle(
    os.path.join(data_dir, "labels_binary_split.pkl"))

# preprocessing text documents
# ===============================================
vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.max_document_length)
x_train = np.array(list(vocab_processor.fit_transform(train_text)))
x_dev = np.array(list(vocab_processor.transform(dev_text)))

print("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))

num_classes = y_binary_train.shape[1]
print("num of classes: {:d}".format(num_classes))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = KimCNN(
            sequence_length=x_train.shape[1],
            num_classes=num_classes,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            loss_function=FLAGS.loss_function)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name),
                                                     tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                               'runs', dataset_id,
                                               'kim_cnn'))
        
        print("Writing to {}\n".format(out_dir))

        if tf.gfile.Exists(out_dir):
            print('cleaning ', out_dir)
            tf.gfile.DeleteRecursively(out_dir)
        tf.gfile.MakeDirs(out_dir)

        # Summaries for loss and precision
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        p1 = tf.summary.scalar("p1", cnn.p1)
        p3 = tf.summary.scalar("p3", cnn.p3)
        p5 = tf.summary.scalar("p5", cnn.p5)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, p1, p3, p5, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, p1, p3, p5])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(data_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_binary_batch, y_batch_labels):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y_binary: y_binary_batch,
              cnn.input_y_labels: label_lists_to_sparse_tuple(
                  y_batch_labels, num_classes),  # needs some conversion
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, p1, p3, p5 = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.p1, cnn.p3, cnn.p5],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, p1 {:g}, p3 {:g}, p5 {:g}".format(
                time_str, step, loss, p1, p3, p5))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_binary_batch, y_batch_labels, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y_binary: y_binary_batch,
              cnn.input_y_labels: label_lists_to_sparse_tuple(
                  y_batch_labels, num_classes),  # needs some conversion
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, p1, p3, p5 = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.p1, cnn.p3, cnn.p5],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("[DEV] {}: step {}, loss {:g}, p1 {:g}, p3 {:g}, p5 {:g}".format(
                time_str, step, loss, p1, p3, p5))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_binary_train, y_id_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_binary_batch, y_id_batch = zip(*batch)
            train_step(x_batch, y_binary_batch, y_id_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_binary_dev, y_id_dev, writer=dev_summary_writer)
                print("")

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

