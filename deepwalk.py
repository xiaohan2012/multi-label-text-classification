# coding: utf-8

# In[2]:

import tensorflow as tf
import os
import time
import itertools
import numpy as np

from data_helpers import RWBatchGenerator
from tf_helpers import save_embedding_for_viz


# In[3]:

tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 5000)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")


# In[4]:

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# In[6]:

data_dir = FLAGS.data_dir
metadata_path = '/home/cloud-user/code/network_embedding/{}/labels_for_visualization.tsv'.format(data_dir)

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 3       # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.

walks = RWBatchGenerator.read_walks("{}/random_walks.txt".format(data_dir))

vocabulary_size = len(set(itertools.chain(*walks)))

generator = RWBatchGenerator(walks, batch_size, num_skips, skip_window)

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
num_sampled = 64    # Number of negative examples to sample.



from word2vec import Word2Vec
graph = tf.Graph()
with graph.as_default():
    model = Word2Vec(num_sampled,
                     vocabulary_size=vocabulary_size, embedding_size=embedding_size)

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(model.loss)

    # Add variable initializer.
    init = tf.global_variables_initializer()


# In[8]:

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "deepwalk"))
#                                       "deepwalk-{}".format(timestamp)))
if tf.gfile.Exists(out_dir):
    tf.gfile.DeleteRecursively(out_dir)
tf.gfile.MakeDirs(out_dir)

print("Writing to {}\n".format(out_dir))

# summary config
loss_summary = tf.summary.scalar("loss", model.loss)
train_summary_op = tf.summary.merge([loss_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")

# checkpoint config
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# In[ ]:

# Step 5: Begin training.
num_steps = 9999999

with tf.Session(graph=graph) as session:
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generator.next_batch()
        
        feed_dict = {model.train_inputs: batch_inputs,
                     model.train_labels: np.expand_dims(np.array(batch_labels), -1)}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val, summaries = session.run([optimizer, model.loss, train_summary_op], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        if step % FLAGS.checkpoint_every == 0 and step > 0:
            path = saver.save(session, checkpoint_prefix, global_step=step)
            
            print("Saved model checkpoint to {}\n".format(path))
            
            save_embedding_for_viz(model.embeddings, session, metadata_path, checkpoint_dir)
            
        train_summary_writer.add_summary(summaries, step)
       
