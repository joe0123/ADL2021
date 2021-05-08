export CUDA_VISIBLE_DEVICES='1'
#python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train_valid -s 0.2
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name bert-base-chinese
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext-large
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name bert-base-chinese --epoch 3 --grad_accum_steps 8 --lr 3e-5
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext --epoch 3 --grad_accum_steps 8 --lr 3e-5
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-macbert-base --epoch 3 --grad_accum_steps 8 --lr 3e-5

#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --config_name bert-base-chinese --tokenizer_name bert-base-chinese --lr 5e-5 --epoch 8 --sched_type constant

python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train -s 0
#python train.py --train_file task_data/train_0.json --model_name bert-base-chinese
#python train.py --train_file task_data/train_0.json --model_name hfl/chinese-roberta-wwm-ext 
python train.py --train_file task_data/train_0.json  --model_name hfl/chinese-roberta-wwm-ext-large
#python train.py --train_file task_data/train_0.json --model_name hfl/chinese-roberta-wwm-ext --epoch 3 --grad_accum_steps 8 --lr 3e-5
#python train.py --train_file task_data/train_0.json --model_name hfl/chinese-macbert-base --epoch 3 --grad_accum_steps 8 --lr 3e-5

#python train.py --train_file task_data/train_0.json --config_name bert-base-chinese --tokenizer_name bert-base-chinese --lr 5e-5 --epoch 8 --sched_type constant

#python make_data.py -q ../dataset/public.json -c ../dataset/context.json -o public
#python predict.py --test_file task_data/public_0.json --target_dir saved/roberta_large
