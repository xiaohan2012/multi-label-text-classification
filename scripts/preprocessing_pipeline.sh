#! /bin/zsh

python build_question_graph.py --data_dir $1
python process_posts.py --data_dir $1
python extract_X_and_Y.py --data_dir $1
