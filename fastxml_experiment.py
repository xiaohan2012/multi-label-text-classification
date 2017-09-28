# coding: utf-8

# supprese warning
import pickle as pkl
import numpy as np
import tensorflow as tf
import os
import warnings


def warn(*args, **kwargs):
    pass

warnings.warn = warn

from fastxml import Trainer, Inferencer

from eval_helpers import precision_at_ks


tf.flags.DEFINE_string('data_dir', 'data/datascience/', 'directory of dataset')
tf.flags.DEFINE_integer('n_trees', 32, 'number of forests')
tf.flags.DEFINE_boolean('eval', False, "whether evaluate on test or not")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_dir = FLAGS.data_dir

# load train/test data
x_train, x_dev, x_test = pkl.load(open(os.path.join(data_dir, "tfidf_split.pkl"), 'rb'))
y_train, y_dev, y_test = pkl.load(open(os.path.join(data_dir, "labels_id_split.pkl"), 'rb'))

# convert dtype to be compatible with fastxml
x_train.data = np.asarray(x_train.data, dtype=np.float32)
x_dev.data = np.asarray(x_dev.data, dtype=np.float32)
x_test.data = np.asarray(x_test.data, dtype=np.float32)

# fastxml
model_path = os.path.join(data_dir, 'fastxml.model')
     
if not FLAGS.eval:
    print("training...")
    trainer = Trainer(n_trees=FLAGS.n_trees, n_jobs=-1)
    trainer.fit(list(x_train), y_train)
    trainer.save(model_path)

clf = Inferencer(model_path)


ks = [1, 3, 5]
print("fastxml:")

if not FLAGS.eval:
    print('validating...')
    pred = clf.predict(x_dev)
    precs = precision_at_ks(pred, y_dev, ks=ks)
else:
    print('testing...')
    pred = clf.predict(x_test)
    precs = precision_at_ks(pred, y_test, ks=ks)

print("{} result".format("Test" if FLAGS.eval else "Dev"))
for p, k in zip(precs, ks):
    print("p@{}: {:.2f}".format(k, p))
