#python3 train.py --task intent --embed_lr 1e-5 --sched_name cosine
python3 train.py --task intent --embed_lr 0 --sched_name const --device cuda:1
