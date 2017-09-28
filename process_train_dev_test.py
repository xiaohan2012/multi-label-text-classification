# coding: utf-8
"""
encode:
1. the text using tf-idf
2. labels into binary vector
"""

import os
import pickle as pkl
import itertools
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer

from data_helpers import MultiLabelIntegerEncoder, label_ids_to_binary_matrix


tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')
tf.flags.DEFINE_integer('tag_freq_threshold', 0, 'minimum frequency of a tag')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_dir = FLAGS.data_dir

dump_path = '{}/train_dev_test.pkl'.format(data_dir)
(x_text_train, x_text_dev, x_text_test,
 y_labels_train, y_labels_dev, y_labels_test,
 node_ids_train, node_ids_dev, node_ids_test) = \
                                                pkl.load(open(dump_path, 'rb'))


vectorizer = TfidfVectorizer()
x_tfidf_train = vectorizer.fit_transform(x_text_train)
x_tfidf_dev, x_tfidf_test = vectorizer.transform(x_text_dev), vectorizer.transform(x_text_test)


label_encoder = MultiLabelIntegerEncoder()


def labels_to_str_list(y_labels):
    return list(map(lambda s: s.split(','), y_labels))


y_ints_train = label_encoder.fit_transform(labels_to_str_list(y_labels_train))
y_ints_dev, y_ints_test = label_encoder.transform(y_labels_dev), \
                          label_encoder.transform(y_labels_test)

n_cols = len(set(itertools.chain(*y_ints_train)))

y_binary_train = label_ids_to_binary_matrix(y_ints_train, (len(y_ints_train), n_cols))
y_binary_dev = label_ids_to_binary_matrix(y_ints_dev, (len(y_ints_dev), n_cols))
y_binary_test = label_ids_to_binary_matrix(y_ints_test, (len(y_ints_test), n_cols))

text_path = os.path.join(data_dir, 'text_split.pkl')
tfidf_path = os.path.join(data_dir, 'tfidf_split.pkl')
labels_path = os.path.join(data_dir, 'labels_split.pkl')
labels_id_path = os.path.join(data_dir, 'labels_id_split.pkl')
labels_binary_path = os.path.join(data_dir, 'labels_binary_split.pkl')
node_ids_path = os.path.join(data_dir, 'node_ids_split.pkl')


label_encoder_path = os.path.join(data_dir, 'label_encoder.pkl')
tfidf_vectorizer_path = os.path.join(data_dir, 'text_vectorizer.pkl')


def dump_data(variable, path):
    pkl.dump(variable, open(path, 'wb'))


dump_info = [
    ((x_text_train, x_text_dev, x_text_test), text_path),
    ((x_tfidf_train, x_tfidf_dev, x_tfidf_test), tfidf_path),
    ((y_labels_train, y_labels_dev, y_labels_test), labels_path),
    ((y_ints_train, y_ints_dev, y_ints_test), labels_id_path),
    ((y_binary_train, y_binary_dev, y_binary_test), labels_binary_path),
    ((node_ids_train, node_ids_dev, node_ids_test), node_ids_path)
]


for var, path in dump_info:
    print('dumping to', path)
    dump_data(var, path)
