# network_embedding

# important scripts

- `process_posts.py`: extract text and labels from raw csv file downloaded from stackexchange
- `extract_X_and_Y.py`: extract tfidf features and encode labels from text
- `sample_random_walks.py`: sample random walks on a graph
- `build_question_graph.py`: extract largest CC in graph and dump it (including the vertex-question id mapping)

# pipeline

run:

1. `build_question_graph.py`: get the connected question ids and question graph for graph embedding
2. `process_posts.py`: build the `input_text.csv` and `labels.csv`
3. `extract_X_and_Y.py`: tfidf on input texts (for `fastxml`)