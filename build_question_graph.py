# coding: utf-8

import pandas as pd
import numpy as np
import itertools
import pickle as pkl

from graph_tool import Graph
from graph_tool.topology import label_largest_component
from collections import defaultdict
from scipy import sparse as sp


QUESTION = 1

data_dir = 'data/stackexchange/datascience'
df = pd.read_csv('{}/posts.csv'.format(data_dir), sep=',')

# create a graph
# each node is a question,
# a question is associated with a list of users, including the author of both the question and answers

# question to users mapping
q2us = defaultdict(set)

for i, r in df.iterrows():
    pid = None
    if r['PostTypeId'] == QUESTION:
        pid = int(r['Id'])
    else:
        parend_id = r['ParentId']
        if parend_id > 0:
            pid = int(parend_id)

    if pid:
        uname, uid = r['OwnerDisplayName'], r['OwnerUserId']
        if not np.isnan(uid):
            q2us[pid].add(int(uid))
        elif isinstance(uname, str):
            q2us[pid].add(uname)


id2q_map = dict(enumerate(q2us))
q2id_map = dict(zip(id2q_map.values(), id2q_map.keys()))


all_users = set(itertools.chain(*q2us.values()))
id2u_map = dict(enumerate(all_users))
u2id_map = dict(zip(id2u_map.values(), id2u_map.keys()))


# create a bi-partite adjacency matrix, row->question, column->user
n_entries = sum(map(len, q2us.values()))
data = np.ones(n_entries)
row_idx = []
col_idx = []
for q, us in q2us.items():
    row_idx += [q2id_map[q]]*len(us)
    col_idx += [u2id_map[u] for u in us]
assert len(data) == len(row_idx) == len(col_idx)
m = sp.csr_matrix((data, (row_idx, col_idx)), shape=(len(q2id_map), len(u2id_map)))


qm = m * m.T  # question adj matrix via unipartite projection

g = Graph()
edges = zip(*qm.nonzero())
g.add_edge_list(edges)

prop = label_largest_component(g)
f = np.sum(prop.a) / len(prop.a)
print('fraciton of nodes in largest cc: {}'.format(f))

print('saving graph')
g.save('{}/question_graph.gt'.format(data_dir))


print('dumping id mapping')
pkl.dump(
    {'id2q_map': id2q_map, 'q2id_map': q2id_map, 'id2u_map': id2u_map, 'u2id_map': u2id_map},
    open('{}/question_id_mapping.pkl'.format(data_dir), 'wb'))
