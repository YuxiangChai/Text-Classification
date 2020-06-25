# Text Classification

## Requirements

- Pytorch >= 10.1

- Python >= 3.6

## Dataset

DBpedia with following format

```
__label__1 some text some text here.
```

## Train and test

```
$ python main.py --model model_type --train_file path/to/train --test_file path/to/test
```

- **model_type**: `fasttext` or `cnn` or `lstm`