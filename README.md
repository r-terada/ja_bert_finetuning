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

|model(max_len)|macro-f1|weighted-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.8173|0.8559|0.8590|80.9250|
|cl-tohoku/bert-base-japanese-v2(512)|0.8438|0.8697|0.8716|33.4410|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8150|0.8444|0.8499|81.0240|
|cl-tohoku/bert-large-japanese|0.8026|0.8424|0.8463|26.7780|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|**0.8594**|**0.8889**|**0.8897**|80.4180|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator(512)|0.8530|0.8793|0.8807|33.2370|
|nlp-waseda/roberta-base-japanese|0.8496|0.8719|0.8770|81.7610|
|nlp-waseda/roberta-base-japanese(512)|**0.8576**|**0.8832**|**0.8843**|33.4360|
|izumi-lab/electra-base-japanese-discriminator|0.8116|0.8422|0.8463|80.2210|
|chiTra-1|0.8214|0.8594|0.8608|77.5350|

### jrte-entailment

script: entailment.py

|model(maxlen)|macro-f1|weighted-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.8951|0.9023|0.9024|81.8640|
|cl-tohoku/bert-base-japanese-v2(512)|**0.8959**|**0.9035**|**0.9042**|33.4770|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8938|0.9016|0.9024|79.6210|
|cl-tohoku/bert-large-japanese|0.9017|0.9089|0.9096|26.7530|
|cl-tohoku/bert-large-japanese(512, batch_size=16)|0.8846|0.8928|0.8933|8.0640|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8915|0.8987|0.8987|79.6810|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator(512)|**0.8936**|**0.9015**|**0.9024**|33.4070|
|nlp-waseda/roberta-base-japanese|**0.8984**|**0.9056**|**0.9060**|82.0670|
|nlp-waseda/roberta-base-japanese(512)|0.8933|0.9013|0.9024|33.4800|
|izumi-lab/electra-base-japanese-discriminator|0.8917|0.8997|0.9005|79.1110|
|chiTra-1|**0.9054**|**0.9124**|**0.9132**|78.5920|
|chiTra-1(512)|**0.9052**|**0.9123**|**0.9132**|32.7880|

