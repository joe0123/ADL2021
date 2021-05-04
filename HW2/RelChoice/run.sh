export CUDA_VISIBLE_DEVICES='2'
#python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train_valid -s 0.2
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-xlnet-base
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-bert-wwm-ext
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name bert-base-chinese
#CUDA_VISIBLE_DEVICES='2' python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --config_name bert-base-chinese --tokenizer_name bert-base-chinese --lr 5e-5 --sched_type constant --epoch 10

#python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train -s 0
#python train.py --train_file task_data/train_0.json --model_name hfl/chinese-xlnet-base
#python train.py --train_file task_data/train_0.json --model_name hfl/chinese-roberta-wwm-ext
#python train.py --train_file task_data/train_0.json --model_name hfl/chinese-bert-wwm-ext
#python train.py --train_file task_data/train_0.json --model_name bert-base-chinese
#python train.py --train_file task_data/train_0.json --config_name saved/q4/config.json --tokenizer_name bert-base-chinese --epoch_num 5 --lr 1e-4
#CUDA_VISIBLE_DEVICES='2' python train.py --train_file task_data/train_0.json --config_name bert-base-chinese --tokenizer_name bert-base-chinese --lr 5e-5 --sched_type constant --epoch 5


python make_data.py -q ../dataset/public.json -c ../dataset/context.json -o public
python predict.py --raw_test_file ../dataset/public.json --test_file task_data/public_0.json --target_dir saved/bert
