python make_data.py -q ../dataset/train.json -c ../dataset/context.json -o train -s 0.2
python train.py --train_file task_data/train_0.json --valid_file task_data/train_1.json --pretrained_name hfl/chinese-xlnet-base
