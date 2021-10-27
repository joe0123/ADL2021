# ADL HW0 - Text sentiment classification
## Task description

A binary classification problem for sentiment analysis in Chinese
* https://docs.google.com/presentation/d/1CqYNZFZDhn-d5ZD30zjw_O36CQUwSOemwwXnTLG07RE/
* https://www.kaggle.com/c/ntu-adl-hw0-spring-2021

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
|  Public Test | 0.93300 (1/90) | 0.91080 (10/90) |
| Private Test | 0.92840 (1/90) | 0.91520 (2/90)  |
