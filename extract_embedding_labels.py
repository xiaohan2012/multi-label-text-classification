# coding: utf-8

import pickle as pkl
import pandas as pd
import tensorflow as tf

from collections import Counter
from itertools import chain


tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

data_dir = FLAGS.data_dir

labels = pd.read_csv('{}/labels.csv'.format(data_dir), header=None, index_col=0)


qids = pkl.load(open('{}/connected_question_ids.pkl'.format(data_dir), 'rb'))


all_occurence = chain(*map(lambda s: s.split(','), labels[1].tolist()))
label_freq = Counter(all_occurence)


labels_to_show = []
for i, r in labels.loc[qids].iterrows():
    best_label = max(r[1].split(','), key=label_freq.__getitem__)
    labels_to_show.append(best_label)

output_path = '{}/labels_for_visualization.tsv'.format(data_dir)

print('save to {}'.format(output_path))

with open(output_path, 'w') as f:
    for l in labels_to_show:
        f.write(l + '\n')
