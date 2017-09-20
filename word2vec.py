import math
import tensorflow as tf


class Word2Vec():
    """
    model for word2vec
    can be used network embedding as well

    Args:

    num_sampled: int, number of negative examples to sample
    vocabulary_size: int
    embedding_size: int
    """

    def __init__(self,
                 num_sampled,
                 vocabulary_size,
                 embedding_size):

        self.vocabulary_size, self.embedding_size = (vocabulary_size,
                                                     embedding_size)
        assert self.vocabulary_size > 0
        assert self.embedding_size > 0

        # Input data.
        self.train_inputs = tf.placeholder(tf.int32, shape=None)
        self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up self.embeddings for inputs.
            with tf.name_scope('embedding'):
                self.embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            with tf.name_scope('nce'):
                # Construct the variables for the NCE loss
                self.nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))
                self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nce_weights,
                               biases=self.nce_biases,
                               labels=self.train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=self.vocabulary_size))

        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
