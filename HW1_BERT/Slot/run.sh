export CUDA_VISIBLE_DEVICES='2'
python make_data.py -i ../datasets/slot/train.json -o train -d task_data
python make_data.py -i ../datasets/slot/eval.json -o valid -d task_data
python train.py --train_file task_data/train.json --valid_file task_data/valid.json --model_name bert-base-uncased

#python make_data.py -i ../datasets/intent/train.json ../datasets/intent/eval.json -o all -d task_data
#python train.py --train_file task_data/all.json --model_name bert-base-uncased


#python make_data.py -i ../datasets/intent/test.json -o test -d task_data
#python predict.py --test_file task_data/test.json --target_dir saved/bert
