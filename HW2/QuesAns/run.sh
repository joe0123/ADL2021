#python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train_valid -s 0.2
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name bert-base-chinese
#CUDA_VISIBLE_DEVICES='2' python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext
#CUDA_VISIBLE_DEVICES='1' python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext-large
#CUDA_VISIBLE_DEVICES='1' python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name bert-base-chinese --epoch 3 --grad_accum_steps 8 --lr 3e-5
#CUDA_VISIBLE_DEVICES='0' python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext --epoch 3 --grad_accum_steps 8 --lr 3e-5
#CUDA_VISIBLE_DEVICES='0' python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-macbert-base --epoch 3 --grad_accum_steps 8 --lr 3e-5

#CUDA_VISIBLE_DEVICES='1' python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --config_name bert-base-chinese --tokenizer_name bert-base-chinese --lr 1e-4 --epoch 10 --sched_type constant

#python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train -s 0
#python train.py --train_file task_data/train_0.json --model_name bert-base-chinese
#python train.py --train_file task_data/train_0.json --model_name hfl/chinese-roberta-wwm-ext 
#CUDA_VISIBLE_DEVICES='2' python train.py --train_file task_data/train_0.json  --model_name hfl/chinese-roberta-wwm-ext-large
#CUDA_VISIBLE_DEVICES='0' python train.py --train_file task_data/train_0.json --model_name hfl/chinese-roberta-wwm-ext --epoch 3 --grad_accum_steps 8 --lr 3e-5
#CUDA_VISIBLE_DEVICES='0' python train.py --train_file task_data/train_0.json --model_name hfl/chinese-macbert-base --epoch 3 --grad_accum_steps 8 --lr 3e-5

python make_data.py -q ../dataset/public.json -c ../dataset/context.json -o public
CUDA_VISIBLE_DEVICES='2' python predict.py --test_file task_data/public_0.json --target_dir saved/roberta_large
