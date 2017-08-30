# coding: utf-8

import pickle as pkl
import numpy as np
import tensorflow as tf
import os

from sklearn.cross_validation import train_test_split

from fastxml import Trainer, Inferencer
from fastxml.weights import propensity

from eval_helpers import precision_at_ks



tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')
tf.flags.DEFINE_integer('tag_freq_threshold', 5, 'minimum frequency of a tag')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_dir = FLAGS.data_dir

# load train/test data
X = pkl.load(open(os.path.join(data_dir, "X.pkl"), 'rb'))
Y = pkl.load(open(os.path.join(data_dir, "Y.pkl"), 'rb'))

# convert dtype to be compatible with fastxml
X.data = np.asarray(X.data, dtype=np.float32)

# split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)


# fastxml
model_path = os.path.join(data_dir, 'fastxml.model')

trainer = Trainer(n_trees=32, n_jobs=-1)
trainer.fit(list(X_train), Y_train)
trainer.save(model_path)

clf = Inferencer(model_path)
pred = clf.predict(X_test)

print("fastxml:")
precision_at_ks(pred, Y_test)


# PFastreXML
model_path = os.path.join(data_dir, 'pfastrexml.model')

weights = propensity(Y_train)

trainer = Trainer(n_trees=32, n_jobs=-1, leaf_classifiers=True)

trainer.fit(list(X_train), Y_train, weights)

trainer.save(model_path)

clf = Inferencer(model_path)

pred = clf.predict(X_test)

print("PFastreXML:")
precision_at_ks(pred, Y_test)
