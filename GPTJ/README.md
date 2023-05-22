# GPTJ

This folder containing the code for finetuning the model with LoRa.
Install all dependencies by running

```
pip install -r requirements.txt
```
## Step 1: Fine tune the model 

To fine tune the model run


```
./run_finetune.sh
```

This runs the `finetune.py` script with the following arguments:
- seed 42
- model_name EleutherAI/gpt-j-6B
- toknizer_name EleutherAI/gpt-j-6B
- max_seq_length 256
- batch_size 4
- num_train_epochs 10
- warmup 0.1
- learning_rate 5e-6
- input_text_path pseudoCode-Dataset/pseudoCode_csv_full/
- save gptj-6B

## Step 2: Train SyRELM model 

```
./run_RL.sh
```

This runs the `RL.py` script with the following arguments:
- seed 42
- model_name EleutherAI/gpt-j-6B
- toknizer_name EleutherAI/gpt-j-6B
- max_seq_length 256
- stateDict_path gptj-6B
- batch_size 4
- mini_batch_size 1
- num_train_epochs 10
- warmup 0.1
- learning_rate 1.41e-6
- init_kl_coef 0.03
- gamma 0.99
- input_text_path pseudoCode-Dataset/pseudoCode_csv_full/
- save gptj-6B
  
To run a model for inference run on a dataset:

```
python3 utils/generate.py
```

The pseudo code dataset is in `data/pseudoCode-Dataset/pseudoCode_csv_full_filteredV2`

Dataset is curriculum format is also present in `data/pseudoCode-Dataset/pseudoCode_curriculum`
