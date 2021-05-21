export CUDA_VISIBLE_DEVICES='1'
#python train.py --train_file data/train_split.jsonl --valid_file data/valid_split.jsonl --model_name google/mt5-small
#python train.py --train_file data/train_split.jsonl --valid_file data/valid_split.jsonl --model_name google/mt5-base


#python train.py --train_file data/train.jsonl --model_name google/mt5-small
#python train.py --train_file data/train.jsonl --model_name google/mt5-base

python predict.py --test_file data/valid_split.jsonl --target_dir saved/mt5_small_val
#python predict.py --test_file data/public.jsonl --target_dir saved/mt5_small
#python predict.py --test_file data/public.jsonl --target_dir saved/mt5_base
