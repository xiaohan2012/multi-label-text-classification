# coding: utf-8

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
    nodes = np.arange(g.num_vertices())
    while n > 0:
        if n >= g.num_vertices():
            nodes_to_start = np.arange(g.num_vertices())
        else:
            nodes_to_start = np.random.choice(nodes, n, replace=False)

        for v in nodes_to_start:
            yield random_walk(g, v, walk_length, alpha)
            
        n -= len(nodes_to_start)


if __name__ == '__main__':
    data_dir = 'data/stackexchange/datascience/'

    # params on random walk
    num_walks = 10000
    walk_length = 6

    # output
    walks = []  # list of list of nodes

    g = load_graph('{}/question_graph.gt'.format(data_dir))
    total = 6095*2
    with open('{}/random_walks.txt'.format(data_dir), 'w') as f:
        for walk in tqdm(yield_n_random_walks(total, g, 10, 0.05), total=total):
            f.write(' '.join(map(str, walk)) + '\n')
