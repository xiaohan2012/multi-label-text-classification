# network_embedding

# important scripts

- `process_posts.py`: extract text and labels from raw csv file downloaded from stackexchange
- `extract_X_and_Y.py`: extract tfidf features and encode labels from text
- `sample_random_walks.py`: sample random walks on a graph
- `build_question_graph.py`: extract largest CC in graph and dump it (including the vertex-question id mapping)
