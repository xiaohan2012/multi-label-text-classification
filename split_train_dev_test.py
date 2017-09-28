import os
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split


tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')
tf.flags.DEFINE_float('test_sample_percentage', 0.1, 'precentage for test samples')
tf.flags.DEFINE_float('dev_sample_percentage', 0.1, 'precentage for dev samples')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

data_dir = FLAGS.data_dir

text_path = os.path.join(data_dir, "input_text.csv")
tdf = pd.read_csv(text_path, header=None)
x_text = tdf[1]

label_path = os.path.join(data_dir, "labels.csv")
ldf = pd.read_csv(label_path, header=None)
y_labels = ldf[1]

node_ids = np.arange(len(x_text))

# train+dev and test
(x_text_train, x_text_test,
 y_labels_train, y_labels_test,
 node_ids_train, node_ids_test) = \
                                  train_test_split(
                                      x_text, y_labels, node_ids,
                                      train_size=1 - FLAGS.test_sample_percentage,
                                      random_state=123456)

# re-scale
train_percentage = 1 - FLAGS.dev_sample_percentage - FLAGS.test_sample_percentage
new_train_percentage = train_percentage / (train_percentage + FLAGS.dev_sample_percentage)

# train and dev
(x_text_train, x_text_dev,
 y_labels_train, y_labels_dev,
 node_ids_train, node_ids_dev) = \
                                  train_test_split(
                                      x_text_train, y_labels_train, node_ids_train,
                                      train_size=new_train_percentage,
                                      random_state=42)

print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(
    len(x_text_train), len(x_text_dev), len(x_text_test)))

dump_path = '{}/train_dev_test.pkl'.format(data_dir)

print('dumping to ', dump_path)
pkl.dump((x_text_train.tolist(), x_text_dev.tolist(), x_text_test.tolist(),
          y_labels_train.tolist(), y_labels_dev.tolist(), y_labels_test.tolist(),
          node_ids_train, node_ids_dev, node_ids_test),
         open(dump_path, 'wb'))
