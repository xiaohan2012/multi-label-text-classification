# coding: utf-8
"""
encode:
1. the text using tf-idf
2. labels into binary vector
"""

import os
import pickle as pkl
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer

from data_helpers import MultiLabelIntegerEncoder


tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')
tf.flags.DEFINE_integer('tag_freq_threshold', 5, 'minimum frequency of a tag')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_dir = FLAGS.data_dir


label_path = os.path.join(data_dir, "labels.csv")
text_path = os.path.join(data_dir, "input_text.csv")

vectorizer = TfidfVectorizer()


tdf = pd.read_csv(text_path, header=None)
text = tdf[1]
X = vectorizer.fit_transform(text)


ldf = pd.read_csv(label_path, header=None)
labels = ldf[0]


label_encoder = MultiLabelIntegerEncoder()
Y = label_encoder.fit_transform(labels.apply(lambda s: s.split(',')).tolist())

pkl.dump(label_encoder, open(os.path.join(data_dir, 'label_encoder.pkl'), 'wb'))
pkl.dump(vectorizer, open(os.path.join(data_dir, 'text_vectorizer.pkl'), 'wb'))
pkl.dump(X, open(os.path.join(data_dir, 'X.pkl'), 'wb'))
pkl.dump(Y, open(os.path.join(data_dir, 'Y.pkl'), 'wb'))

