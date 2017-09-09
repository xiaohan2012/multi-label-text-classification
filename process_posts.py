# coding: utf-8

import os
import re
import pandas as pd
import itertools
import tensorflow as tf
from data_helpers import strip_tags, clean_str


tf.flags.DEFINE_string('data_dir', '', 'directory of dataset')
tf.flags.DEFINE_integer('tag_freq_threshold', 5, 'minimum frequency of a tag')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

data_dir = FLAGS.data_dir
tag_freq_threshold = FLAGS.tag_freq_threshold


label_path = os.path.join(data_dir, "labels.csv")
text_path = os.path.join(data_dir, "input_text.csv")
df = pd.read_csv('{}/posts.csv'.format(data_dir), sep=',')


print("dataset containing {} records".format(df.shape[0]))


qs = df[df['PostTypeId'] == 1]  # we only consider questions here

print("contains {} questions".format(qs.shape[0]))

# extract tags
regexp = re.compile("<(.+?)>")


def extract_tags(s):
    return regexp.findall(s)

tags = qs["Tags"].apply(extract_tags).tolist()

# filter out infrequent tags
tag_freq = pd.Series(list(itertools.chain(*tags))).value_counts()
valid_tags = tag_freq.index[tag_freq > tag_freq_threshold]
tag_set = set(valid_tags)


print('number of unique labels (frequency>{}): {}'.format(
    tag_freq_threshold, len(tag_set)))


normalized_tags = [[t for t in ts if t in tag_set] for ts in tags]


# save labels to file
y = pd.Series(list(map(lambda l: ",".join(l), normalized_tags)))

mask = (y.apply(len) > 0)

y = y[mask]
qs.index = mask.index
qs = qs[mask]

assert y.shape[0] == qs.shape[0]

print('num. questions with at least one valid labels: {}'.format(qs.shape[0]))

print('saving labels to {}'.format(label_path))
y.to_csv(label_path, index=False)


body = qs['Body'].apply(strip_tags).apply(clean_str)
title = qs['Title'].apply(strip_tags).apply(clean_str)

# concatenate the texts
input_text = pd.Series([' '.join(l) for l in list(zip(title, body))])


print("saving input text to {}".format(text_path))
input_text.to_csv(text_path)

