# 実験結果

## ハイパーパラメータ

|task|model|batch_size|initial_learning_rate|max_epoch|max_length|
|----|-----|----------|---------------------|---------|----------|
|jrte-sentiment|electra|32|1e-4|10|128|
|jrte-sentiment|other|32|5e-5|10|128|
|jrte-entailment|electra|32|1e-4|10|128|
|jrte-entailment|other|32|5e-5|10|128|

## 数値評価


### jrte-sentimenft

script: jrte/sentiment.py

|model(max_len)|macro-f1|micro-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.8173|0.8590|0.8590|80.9250|
|cl-tohoku/bert-base-japanese-v2(512)|0.8438|0.8716|0.8716|33.4410|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8150|0.8499|0.8499|81.0240|
|cl-tohoku/bert-large-japanese|0.8026|0.8463|0.8463|26.7780|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|**0.8594**|**0.8897**|**0.8897**|80.4180|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator(512)|0.8530|0.8807|0.8807|33.2370|
|nlp-waseda/roberta-base-japanese|0.8496|0.8770|0.8770|81.7610|
|nlp-waseda/roberta-base-japanese(512)|__0.8576__|__0.8843__|__0.8843__|33.4360|
|izumi-lab/electra-base-japanese-discriminator|0.8116|0.8463|0.8463|80.2210|

### jrte-entailment

script: entailment.py

|model|macro-f1|micro-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.8951|0.9024|0.9024|81.8640|
|cl-tohoku/bert-base-japanese-v2(512)|__0.8959__|__0.9042__|__0.9042__|33.4770|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8938|0.9024|0.9024|79.6210|
|cl-tohoku/bert-large-japanese|__0.9017__|__0.9096__|__0.9096__|26.7530|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8915|0.8987|0.8987|79.6810|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator(512)|0.8936|0.9024|0.9024|33.4070|
|nlp-waseda/roberta-base-japanese|**0.8984**|**0.9060**|**0.9060**|82.0670|
|nlp-waseda/roberta-base-japanese(512)|0.8933|0.9024|0.9024|33.4800|
|izumi-lab/electra-base-japanese-discriminator|0.8917|0.9005|0.9005|79.1110|
