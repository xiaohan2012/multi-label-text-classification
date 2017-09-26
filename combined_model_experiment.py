# coding: utf-8

import os
import pickle as pkl
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
import itertools


from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import learn
from itertools import repeat, chain

from kim_cnn import KimCNN
from word2vec import Word2Vec
from combined import Combined
from eval_helpers import label_lists_to_sparse_tuple
from data_helpers import batch_iter, RWBatchGenerator
from tf_helpers import get_variable_value_from_checkpoint
                
from tensorflow.python import debug as tf_debug
from tf_helpers import save_embedding_for_viz


# In[2]:


tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')
tf.flags.DEFINE_integer('tag_freq_threshold', 5, 'minimum frequency of a tag')

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("max_document_length", 2000, "Maximum length of document, exceeding part is truncated")

# Architecutural parameters for KimCNN

tf.flags.DEFINE_string("loss_function", 'sigmoid', "loss function: (softmax|sigmoid) (Default: sigmoid)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")


tf.flags.DEFINE_integer("dw_batch_size", 128, "Batch Size for deep walk model (default: 128)")
tf.flags.DEFINE_integer("dw_skip_window", 3, "How many words to consider left and right. (default: 3)")
tf.flags.DEFINE_integer("dw_num_skips", 4, "How many times to reuse an input to generate a label. (default: 4)")
tf.flags.DEFINE_integer("dw_embedding_size", 128, "Dimensionality of node embedding. (default: 128)")
tf.flags.DEFINE_integer("dw_num_negative_samples", 64, "Number of negative examples to sample. (default: 64)")


# global training parameter
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

tf.flags.DEFINE_string("pretrained_embedding_checkpoint_dir", "",
                       "directory of checkpoint where pretrained embedding lives")
tf.flags.DEFINE_string("pretrained_embedding_name",
                       "embedding/table",
                       "variable name of the pretrained emebdding (defualt: embedding/table)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# In[5]:


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

data_dir = FLAGS.data_dir


# In[6]:


# load text data and label information

text_path = os.path.join(data_dir, "input_text.csv")
tdf = pd.read_csv(text_path, header=None)
x_text = tdf[1]


vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.max_document_length)
X = np.array(list(vocab_processor.fit_transform(x_text)))
node_ids = np.arange(X.shape[0])

# load train/test data
Y_labels = pkl.load(open(os.path.join(data_dir, "Y.pkl"), 'rb'))


size = sum(len(ls) for ls in Y_labels)
row_indx = list(chain(*[list(repeat(i, len(ls))) for i, ls in enumerate(Y_labels)]))
col_indx = list(chain(*Y_labels))
Y_binary = csr_matrix((np.ones(size), (row_indx, col_indx)),
                      shape=(len(Y_labels), len(set(col_indx)))).toarray()


# split data
x_train, x_dev, y_train_binary, y_dev_binary, y_train_labels, y_dev_labels, train_node_ids, dev_node_ids = train_test_split(
    X, Y_binary, Y_labels, node_ids, train_size=1 - FLAGS.dev_sample_percentage, random_state=42)
print("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))

num_classes = y_train_binary.shape[1]
print("num of classes: {:d}".format(num_classes))


# In[7]:


# load node embedding data

walks = RWBatchGenerator.read_walks("{}/random_walks.txt".format(data_dir))

vocabulary_size = len(set(itertools.chain(*walks)))

dw_data_generator = RWBatchGenerator(
    walks, FLAGS.dw_batch_size, FLAGS.dw_num_skips, FLAGS.dw_skip_window)


# In[8]:


# Training
# ==================================================


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    # DEBUG
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    with sess.as_default():
        with tf.name_scope('kim_cnn'):
            cnn = KimCNN(
                sequence_length=x_train.shape[1],
                num_classes=num_classes,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                loss_function=FLAGS.loss_function,
                redefine_output_layer=True)

        with tf.name_scope('dw'):
            if FLAGS.pretrained_embedding_checkpoint_dir:
                print('use pretrained embedding')
                embedding_value = get_variable_value_from_checkpoint(
                    FLAGS.pretrained_embedding_checkpoint_dir,
                    FLAGS.pretrained_embedding_name)
                dw = Word2Vec(FLAGS.dw_num_negative_samples,
                              vocabulary_size,
                              FLAGS.dw_embedding_size,
                              embedding_value=embedding_value)
            else:
                dw = Word2Vec(FLAGS.dw_num_negative_samples,
                              vocabulary_size,
                              FLAGS.dw_embedding_size)
        
        with tf.name_scope('combined'):
            model = Combined(cnn, dw)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        
        label_train_op = tf.train.AdamOptimizer(1e-3).minimize(model.label_loss)
        graph_train_op = tf.train.GradientDescentOptimizer(1.0).minimize(model.graph_loss)


        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", 'combined'))
        print("Writing to {}\n".format(out_dir))

        if tf.gfile.Exists(out_dir):
            print('cleaning ', out_dir)
            tf.gfile.DeleteRecursively(out_dir)
        tf.gfile.MakeDirs(out_dir)
        
        # Summaries for loss and precision
        label_loss_summary = tf.summary.scalar("label_loss", model.label_loss)
        graph_loss_summary = tf.summary.scalar("graph_loss", model.graph_loss)

        p1 = tf.summary.scalar("p1", model.p1)
        p3 = tf.summary.scalar("p3", model.p3)
        p5 = tf.summary.scalar("p5", model.p5)

        # Train Summaries
        train_summary_op = tf.summary.merge([label_loss_summary, graph_loss_summary,
                                             p1, p3, p5])
        
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([label_loss_summary, graph_loss_summary,
                                           p1, p3, p5])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        
        sess.run(tf.global_variables_initializer())
        
        def train_label_step(x_batch, y_batch_binary, y_batch_labels, node_ids, writer):
            """
            one training step for the label part
            """
            feed_dict = {
              model.cnn.input_x: x_batch,
              model.cnn.input_y_binary: y_batch_binary,
              model.cnn.input_y_labels: label_lists_to_sparse_tuple(
                  y_batch_labels, num_classes),  # needs some conversion
              model.node_ids: node_ids,  # node ids
              model.cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                
              # the following is in vain
              # tf requires all placeholder to be provided some value
      
              model.dw.train_inputs: [0],
              model.dw.train_labels: [[0]],
            }
            _, step, summaries, label_loss, p1, p3, p5 = sess.run(
                [label_train_op, global_step, train_summary_op, model.label_loss, model.p1, model.p3, model.p5],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, label loss {:g}, p1 {:g}, p3 {:g}, p5 {:g}".format(
                time_str, step, label_loss, p1, p3, p5))
            train_summary_writer.add_summary(summaries, step)

        def train_graph_step(x_batch, batch_labels, writer):
            """
            one training step for the graph part
            """
            feed_dict = {
              model.dw.train_inputs: x_batch,
              model.dw.train_labels: np.expand_dims(np.array(batch_labels), -1),
                
              # the following is in vain
              # tf requires all placeholder to be provided some value
              model.cnn.input_x: list(vocab_processor.transform(["asdfkjahdkfhakslfh"])),  # non-sense stuff
              model.cnn.input_y_binary: [[0] * num_classes],  # with no label
              model.cnn.input_y_labels: label_lists_to_sparse_tuple(
                  [[0]], num_classes),  # needs some conversion
              model.node_ids: [0], # node ids
              model.cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                
            }
            _, step, summaries, graph_loss = sess.run(
                [graph_train_op, global_step, train_summary_op, model.graph_loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, graph loss {:g}".format(
                time_str, step, graph_loss))
            writer.add_summary(summaries, step)

            
        def dev_step(x_batch, y_batch_binary, y_batch_labels, node_ids, writer):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              model.cnn.input_x: x_batch,
              model.cnn.input_y_binary: y_batch_binary,
              model.cnn.input_y_labels: label_lists_to_sparse_tuple(
                  y_batch_labels, num_classes),  # needs some conversion
              model.node_ids: node_ids, # node ids                
              model.cnn.dropout_keep_prob: 1.0,
                
              # in vain
              model.dw.train_inputs: [0],
              model.dw.train_labels: [[0]],                
            }
            step, summaries, label_loss, p1, p3, p5 = sess.run(
                [global_step, dev_summary_op, model.label_loss, model.p1, model.p3, model.p5],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, label loss {:g}, p1 {:g}, p3 {:g}, p5 {:g}".format(
                time_str, step, label_loss, p1, p3, p5))
            
            writer.add_summary(summaries, step)

        batches = batch_iter(
            list(zip(x_train, y_train_binary, y_train_labels, train_node_ids)), FLAGS.batch_size, FLAGS.num_epochs)

        for batch in batches:
            # train label part
            x_batch, y_batch_binary, y_train_labels, x_node_ids = zip(*batch)
            train_label_step(x_batch, y_batch_binary, y_train_labels, x_node_ids, train_summary_writer)
            current_step = tf.train.global_step(sess, global_step)  # one step for label training
            
            # train graph part
            batch_inputs, batch_labels = dw_data_generator.next_batch()
            train_graph_step(batch_inputs, batch_labels, train_summary_writer)
            
            current_step = tf.train.global_step(sess, global_step)  # one step for graph training
            
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev_binary, y_dev_labels, dev_node_ids, dev_summary_writer)
                print("")
                
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
            
            global_step += 1
