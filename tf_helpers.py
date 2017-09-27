import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def flatten(t):
    """flatten tensor of any dimension to 1d"""
    return tf.reshape(t, [tf.reduce_prod(t.shape)])


def dynamic_max_k_pool(t, k):
    """
    perform dynamic max-k pooling on t,

    note that:

    1. the 
    2. only supports 1d data for now

    Param:
    -----------
    t: Tensor, 2d (batches, repr)
    k: int

    Return:
    -----------
    Tensor, 2d (batches, k)
    """


def save_embedding_for_viz(embeddings, session, metadata_path, checkpoint_dir):
    embeddings_val = embeddings.eval()

    embedding_var = tf.Variable(embeddings_val,  name='node_embedding')
    session.run(embedding_var.initializer)
    
    # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata_path

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(checkpoint_dir)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)
    
    saver = tf.train.Saver([embedding_var])
    saver.save(session, os.path.join(checkpoint_dir, 'model2.ckpt'), 1)
    print('embedding for visualization saved')


def get_variable_value_from_checkpoint(checkpoint_file, variable_names=[]):
    """load from checkpoint_file and read the values of the variables of `variable_name`
    
    return

    list of variable values
    """
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        vals = []
        for variable_name in variable_names:
            embedding_table = sess.graph.get_operation_by_name(variable_name)
            vals.append(embedding_table.outputs[0].eval())

    return vals
