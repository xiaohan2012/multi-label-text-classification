import tensorflow as tf
from eval_helpers import tf_precision_at_k


class Combined():
    def __init__(self, cnn_model, dw_model):
        self.cnn, self.dw = cnn_model, dw_model

        # input document ids
        # to retrieve node embedding
        self.node_ids = tf.placeholder(dtype=tf.int32, shape=None, name="input_node_ids")
        self.l2_loss = tf.constant(0.0)
        
        self.add_output()
        self.add_losses()
        self.add_performance()

    def add_output(self):
        # redefine output
        # concatenate the filters and node embedding for classification
        with tf.name_scope("output"):
            input_length = self.cnn.num_filters_total + self.dw.embedding_size
            W = tf.get_variable(
                "W",
                shape=[input_length,
                       self.cnn.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.cnn.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            # look up the node embeddings
            node_embeddings = tf.gather(self.dw.normalized_embeddings, self.node_ids)
            input_tensor = tf.concat([self.cnn.h_drop, node_embeddings],
                                     1, name="input_concat")
            self.scores = tf.nn.xw_plus_b(input_tensor, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def add_losses(self):
        # CalculateMean cross-entropy loss        
        with tf.name_scope("loss"):
            if self.cnn.loss_function == 'sigmoid':
                print('use sigmoid xentropy')
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,
                                                                 labels=self.cnn.input_y_binary)
            elif self.cnn.loss_function == 'softmax':
                print('use softmax xentropy')
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,
                                                                 labels=self.cnn.input_y_binary)
            else:
                raise ValueError('invalid loss function')

            # two losses
            self.label_loss = tf.reduce_mean(losses) + self.cnn.l2_reg_lambda * self.l2_loss
            self.graph_loss = self.dw.loss

    def add_performance(self):
        # Accuracy
        with tf.name_scope("performance"):
            self.p1 = tf_precision_at_k(self.scores, self.cnn.input_y_labels, k=1, name='p1')
            self.p3 = tf_precision_at_k(self.scores, self.cnn.input_y_labels, k=3, name='p3')
            self.p5 = tf_precision_at_k(self.scores, self.cnn.input_y_labels, k=5, name='p5')
