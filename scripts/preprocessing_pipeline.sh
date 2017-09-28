#! /bin/zsh

echo "build question graph..."
python build_question_graph.py --data_dir $1

echo "process posts..."
python process_posts.py --data_dir $1

echo "splitting data into train/test/dev..."
python split_train_dev_test.py --data_dir $1

echo "process train dev test..."
python process_train_dev_test.py --data_dir $1
