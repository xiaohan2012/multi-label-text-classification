import tensorflow as tf


class TextCNN:
    def __init__(self, sentence_length, num_classes,
                 embedding_dim,
                 vocab_size,
                 filter_sizes,
                 num_filters, ):
        # placeholders
        self.input_x = tf.placeholder(
            tf.int32, [None, sentence_length], name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

        # variables
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # like name space for the operation inside this context
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -0.1, 1.0))
            embedded_x_3d = tf.nn.embedding_lookup(W, self.input_x)

            # make it into 4 dim
            # equivalent but more verbose way:
            # tf.reshape(embedded_x_3d, [-1, sentence_length, embedding_dim, 1])
            self.embedded_x = tf.expand_dims(embedded_x_3d, -1)

        pooled_outputs = []  # to be concatenated
        for i, filter_size in enumerate(filter_sizes):

            with tf.name_scope('conv-maxpool-{}'.format(filter_size)):
                shape = [filter_size, embedding_dim, 1, num_filters]
                init_W = tf.truncated_normal(shape, stddev=0.1)
                W = tf.Variable(init_W, name='W')

                init_b = tf.constant(0.1, shape=[num_filters])
                b = tf.Variable(init_b, name='b')
                conv = tf.nn.conv2d(self.embedded_x, W,
                                    padding="VALID", strides=[1, 1, 1, 1],
                                    name='conv')
                relu_out = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                maxpool_out = tf.nn.max_pool(
                    relu_out,
                    ksize=[1, (sentence_length - filter_size + 1), 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(maxpool_out)

        num_filters_total = num_filters * len(filter_sizes)

        # 4d: [batch, 1, 1, filter_sizes]
        # "3" here is tricky
        self.h_pool = tf.concat(pooled_outputs, 3)

        # flatten into [batch, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool,
                                      [-1, num_filters_total],
                                      name="pooled_outputs")

        with tf.name_scope('dropout'):
            self.h_dropout = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)
            
        with tf.name_scope('output'):
            W = tf.Variable(
                tf.truncated_normal([num_filters_total, num_classes], stddev=0.1),
                name='W')
            b = tf.Variable(
                tf.constant(0.1, shape=[num_classes]),
                name='b'
            )

            self.scores = tf.nn.xw_plus_b(self.h_dropout, W, b, name="scores")

            probas = tf.sigmoid(self.scores)
            self.predictions = tf.round(probas, name="predictions")

        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('performance'):
            self.precision = tf.metrics.precision(self.input_y, self.predictions, name="precision-micro")[1]
            self.recall = tf.metrics.recall(self.input_y, self.predictions, name="recall-micro")[1]
