# network_embedding

this projects attempts to combine network embedding and convolutional neural network for multi-label text classification.

[read more](https://github.com/xiaohan2012/network_embedding/blob/master/project-slides.pdf)

# utility scripts

- `scripts/preprocessing_pipeline.sh`: all the preprocessing, data splitting, feature extractio, etc
- `sample_random_walks.py`: sample random walks on a graph
- `extract_embedding_labels.py`: extract labels for embedding visualization

# main scripts

- `fastxml_experiment.py`: experiment for fastxml
- `kim_cnn_experiment.py`: experiment for cnn
- `combined_model_experiment.py`: experiment for cnn + deepwalk