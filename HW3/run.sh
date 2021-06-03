export CUDA_VISIBLE_DEVICES='1'
#python train.py --train_file data/train_split.jsonl --valid_file data/valid_split.jsonl --model_name google/mt5-small --epoch 10
#python train.py --train_file data/train_split.jsonl --valid_file data/valid_split.jsonl --model_name saved/mt5_small_val --rl_ratio 1 --epoch_num 5 --lr 3e-5

#python train.py --train_file data/train.jsonl --model_name google/mt5-small --epoch 10
#python train.py --train_file data/train.jsonl --model_name saved/mt5_small --rl_ratio 1 --epoch_num 5 --lr 3e-5

#python predict.py --test_file data/valid_split.jsonl --target_dir saved/rl_val
#python predict.py --test_file data/valid_split.jsonl --target_dir saved/mt5_small --beam_size 5
#python evaluation/eval.py -s results.jsonl -r data/valid_split.jsonl

python predict.py --test_file data/public_.jsonl --target_dir saved/mt5_small --beam_size 5
python evaluation/eval.py -s results.jsonl -r data/public.jsonl


