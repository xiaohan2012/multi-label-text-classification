import re
import itertools
import collections
import numpy as np
import random
from html.parser import HTMLParser


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index: end_index]


class MultiLabelIntegerEncoder:
    def fit(self, labels):
        self.id2label_ = dict(enumerate(set(itertools.chain(*labels))))
        self.label2id_ = dict(zip(self.id2label_.values(), self.id2label_.keys()))

    def transform(self, labels):
        return [[self.label2id_.get(l, -1) for l in ls] for ls in labels]

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)


class RWBatchGenerator():
    """Random walk batch generator
    """
    def __init__(self, walks, batch_size, num_skips, skip_window):
        """
        Args:

        walks: list of integer list
        batch_size: int
        num_skips: int, within each window, number of examples
        skip_window: int, sliding window size
        """
        self.walks = walks
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window

        self.current_walk = 0
        self.data = self.walks[0]

        self.span = 2 * self.skip_window + 1  # [ self.skip_window target self.skip_window ]
        self.data_index = 0
        
    def next_batch(self):
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        buffer = collections.deque(maxlen=self.span)

        if self.data_index + self.span > len(self.data):
            self.data_index = 0

        buffer.extend(self.data[self.data_index:self.data_index + self.span])
        self.data_index += self.span
        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [self.skip_window]
            for j in range(self.num_skips):
                # sample unique targets
                while target in targets_to_avoid:
                    target = random.randint(0, self.span-1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                labels[i * self.num_skips + j] = buffer[target]
            if self.data_index == len(self.data):
                self.current_walk += 1

                if self.current_walk == len(self.walks):  # used all walks
                    self.current_walk = 0
                    
                self.data = self.walks[self.current_walk]

                # equivalent to: buffer[:] = self.data[:self.span]
                buffer.clear()
                buffer.extend(self.data[:self.span])
                self.data_index = self.span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - self.span) % len(self.data)

        return (batch, labels)
    
