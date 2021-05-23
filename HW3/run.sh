export CUDA_VISIBLE_DEVICES='2'
#python train.py --train_file data/train_split.jsonl --valid_file data/valid_split.jsonl --model_name google/mt5-small

#python train.py --train_file data/train.jsonl --model_name google/mt5-small

#python predict.py --test_file data/valid_split.jsonl --target_dir saved/mt5_small_val
#python evaluation/eval.py -s results.jsonl -r data/valid_split.jsonl

#python predict.py --test_file data/public.jsonl --target_dir saved/mt5_small --beam_size 5
python evaluation/eval.py -s results.jsonl -r data/public.jsonl
