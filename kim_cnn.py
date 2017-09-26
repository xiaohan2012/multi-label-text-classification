"""
adapted from:

- https://raw.githubusercontent.com/xiaohan2012/cnn-text-classification-tf/master/text_cnn.py
- Jingzhou, Liu, etc, Deep Learning for Extreme Multi-label Text Classification, SIGIR 2017

"""
import tensorflow as tf
from eval_helpers import tf_precision_at_k


class KimCNN():
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
            loss_function='softmax',
            redefine_output_layer=False):

        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size           = vocab_size
        self.embedding_size       = embedding_size
        self.filter_sizes         = filter_sizes
        self.num_filters          = num_filters
        self.l2_reg_lambda        = l2_reg_lambda
        self.loss_function        = loss_function 
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(
            tf.int32, [None, self.sequence_length], name="input_x")

        # label indicator matrix
        self.input_y_binary = tf.placeholder(
            tf.float32, [None, self.num_classes], name="input_y_binary")

        # label list, a SparseTensor because label list length varies
        self.input_y_labels = tf.sparse_placeholder(
            tf.int32, shape=[None, self.num_classes],
            name='input_y_labels')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        self.l2_loss = tf.constant(0.0)

        # adding the layers
        self.add_embedding_layer()
        self.add_convolution_layer()
        self.add_drop_out()

        if not redefine_output_layer:
            self.add_output()
            self.add_loss()
            self.add_performance()

    def add_embedding_layer(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    def add_convolution_layer(self):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # dim still 4d
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)  # concat by last dim
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

    def add_drop_out(self):
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def add_output(self):
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def add_loss(self):
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            if self.loss_function == 'sigmoid':
                print('use sigmoid xentropy')
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,
                                                                 labels=self.input_y_binary)
            elif self.loss_function == 'softmax':
                print('use softmax xentropy')
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                 labels=self.input_y_binary)
            else:
                raise ValueError('invalid loss function')
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def add_performance(self):
        # Accuracy
        with tf.name_scope("performance"):
            self.p1 = tf_precision_at_k(self.scores, self.input_y_labels, k=1, name='p1')
            self.p3 = tf_precision_at_k(self.scores, self.input_y_labels, k=3, name='p3')
            self.p5 = tf_precision_at_k(self.scores, self.input_y_labels, k=5, name='p5')
