# Fine-tuning results on JGLUE


We ran several fine-tuning experiments on [JGLUE](https://github.com/yahoojapan/JGLUE).  
To use sudachitra as tokenizer for megagonlabs/transformers-ud-japanese-electra-base-discriminator, we added additional diffs to [original patch](https://github.com/yahoojapan/JGLUE/blob/main/fine-tuning/patch/transformers-4.9.2_jglue-1.0.0.patch) and apply it.Modified patch is placed as `transformers-4.9.2_jglue-1.0.0.sudachitra.patch`  
Scripts and commands used in the experiments are the same as in the original
## Hyperparameters 

We searched learning_rate between [1e-4, 2e-5] manually, and epoch is early-stopped watching dev score.

|Model|dataset|epoch|batch_size|learning rate|warmup ratio|max seq length|
|-------|-----|-----|----------|-------------|------------|----------|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|MARC_ja|2|16|3e-5|0.1|512|
||JSTS|4|16|3e-5|0.1|512|
||JNLI|3|16|3e-5|0.1|512|
||JSQuAD|5|16|3e-5|0.1|384|
||JCommonsenseQA|5|16|1e-4|0.1|64|

## Scores 

scores on the JGLUE dev set.

|Model|MARC_ja|JSTS|JNLI|JSQuAD|JCommonsenseQA|
|-----|-------|----|----|------|--------------|
||acc.|Pearson/Spearman|acc.|EM/F1|acc.|
|best score in original baselines|0.964|0.923/0.891|0.924|0.897/0.947|0.901|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.963|0.913/0.877|0.921|0.795/0.887|0.856|

