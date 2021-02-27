# ADL HW0 - Text sentiment classification
## Task description
https://docs.google.com/presentation/d/1CqYNZFZDhn-d5ZD30zjw_O36CQUwSOemwwXnTLG07RE/

## Requirement
* tqdm == 4.56.0
* numpy == 1.20.0
* pandas == 1.2.1
* torch == 1.7.1
* transformers == 4.3.2

## Usage for bert/
Finetune BERT model:
```
python train.py
````
Make inference with finetuned BERT model:
```
python infer.py --ckpt_dir ckpt
```

## Usage for lstm/
Train LSTM model:
```
python train.py
````
Make inference with trained LSTM model:
```
python infer.py --ckpt_dir ckpt
```

## Performance
|              |      BERT      |       LSTM      |
|:------------:|:--------------:|:---------------:|
|     Valid    |     0.92730    |     0.90740     |
|  Public Test | 0.93300 (1/82) | 0.91080 (10/82) |
| Private Test |                |                 |
