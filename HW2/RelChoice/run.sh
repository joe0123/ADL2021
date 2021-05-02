#python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train_valid -s 0.2
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-xlnet-base
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name bert-base-chinese
#python train.py --train_file task_data/train_valid_0.json --valid_file task_data/train_valid_1.json --model_name hfl/chinese-roberta-wwm-ext

python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train -s 0
python train.py --train_file task_data/train_0.json --model_name hfl/chinese-xlnet-base
python train.py --train_file task_data/train_0.json --model_name bert-base-chinese


#python make_data.py -q ../dataset/public.json -c ../dataset/context.json -o public
#python predict.py --raw_test_file ../dataset/public.json --test_file task_data/public_0.json --target_dir saved/xlnet
