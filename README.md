# graph embedding + deep learning for multi-label text classification

this projects attempts to combine:

- **graph embedding**
- **ConNet**

for the purpose of **multi-label text classification**.

I compared three methods on stackexchange datasets, where the goal is to predict the tags of posts.

If you wan to know more, here are [some slides](https://github.com/xiaohan2012/network_embedding/blob/master/project-slides.pdf)

# utility scripts

- `scripts/preprocessing_pipeline.sh`: all the preprocessing, data splitting, feature extractio, etc
- `sample_random_walks.py`: sample random walks on a graph
- `extract_embedding_labels.py`: extract labels for embedding visualization

# main scripts

- `fastxml_experiment.py`: experiment for fastxml
- `kim_cnn_experiment.py`: experiment for cnn
- `combined_model_experiment.py`: experiment for cnn + deepwalk
