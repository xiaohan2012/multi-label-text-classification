# coding: utf-8

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from graph_tool import load_graph


def random_walk(g, start_node, walk_length, alpha=0.05):
    """
    random walk on unweighted, undirected graph
    
    Args:
    alpha: proba of restart
    
    Returns:
    a list of integer list
    """
    walk = [start_node]
    c = g.vertex(start_node)
    for i in range(walk_length):
        if np.random.random() <= alpha:
            n = g.vertex(start_node)
        else:
            n = np.random.choice(list(c.out_neighbours()))
        c = n
        walk.append(int(c))
    return walk


def yield_n_random_walks(n, g, walk_length, alpha):
    nodes = list(map(int, g.vertices()))
    while n > 0:
        if n >= g.num_vertices():
            nodes_to_start = nodes
        else:
            nodes_to_start = np.random.choice(nodes, n, replace=False)

        for v in nodes_to_start:
            yield random_walk(g, v, walk_length, alpha)
            
        n -= len(nodes_to_start)


if __name__ == '__main__':
    tf.flags.DEFINE_string('data_dir', 'data/stackexchange/datascience/', 'directory of dataset')

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    data_dir = FLAGS.data_dir

    # params on random walk
    num_walks_per_node = 80
    window_size = 10
    walk_length = 6
    alpha = 0.05
    
    # output
    walks = []  # list of list of nodes

    g = load_graph('{}/question_graph.gt'.format(data_dir))
    total = num_walks_per_node * g.num_vertices()
    with open('{}/random_walks.txt'.format(data_dir), 'w') as f:
        for walk in tqdm(yield_n_random_walks(total, g, walk_length, alpha), total=total):
            f.write(' '.join(map(str, walk)) + '\n')
    print('written to ', '{}/random_walks.txt'.format(data_dir))
