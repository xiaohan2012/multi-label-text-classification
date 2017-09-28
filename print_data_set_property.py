# coding: utf-8

import itertools
import os
import tensorflow as tf
from data_helpers import load_pickle

tf.flags.DEFINE_string('data_dir', 'data/datascience', 'directory of dataset')


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

data_dir = FLAGS.data_dir
train, dev, test = load_pickle(os.path.join(data_dir,
                                            "labels_id_split.pkl"))


data = train + dev + test

print('#instances: ', len(data))
print('# unique labels: ', len(set(itertools.chain(*data))))
print('avg labels per instance: ', sum(map(len, data)) / len(data))
