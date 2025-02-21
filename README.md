# Install trainind environment similarly with Open-flamingo:
```
conda env create -f environment.yml
```
To install training or eval dependencies, run one of the first two commands. To install everything, run the third command.
```
pip install open-flamingo[training]
pip install open-flamingo[eval]
pip install open-flamingo[all]
```
There are three `requirements.txt` files: 
- `requirements.txt` 
- `requirements-training.txt`
- `requirements-eval.txt`
Depending on your use case, you can install any of these with `pip install -r <requirements-file.txt>`. The base file contains only the dependencies needed for running the model.


# Data conversion
Convert the data from `ADNI_sample` into the MMC4 format using the script `modifications/VLM_ADNI_DATA/convert_adni_to_mmc4.py`. You can modify the script if necessary. The current input for the script is `AD_caption-flamingo_3D_version.json`.

```bash
cd modification/VLM_ADNI_DATA
python convert_adni_to_mmc4.py
```

# Training model
After successfully converting the training data format, run the example training script using the following command:
```bash
torchrun --nnodes=1 --nproc_per_node=8 open_flamingo/train/train.py \
  --lm_path anas-awadalla/mpt-1b-redpajama-200b \
  --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
  --cross_attn_every_n_layers 1 \
  --dataset_resampled \
  --batch_size_mmc4 2 \
  --train_num_samples_mmc4 1000 \
  --workers=4 \
  --run_name OpenFlamingo-3B-vitl-mpt1b \
  --num_epochs 20 \
  --warmup_steps  1875 \
  --mmc4_textsim_threshold 0.24 \
  --mmc4_shards "modifications/VLM_ADNI_DATA/replicate_mmc4/{000000000..000000040}.tar" \
  --report_to_wandb
```

# Evaluation
We have created a modified example for the 3D case with TextVQA evaluation, available at `/demo_eval.sh`. In this version, each question is associated with multiple frames from a 3D visual, instead of a single image as in the original setup.

To adapt the evaluation process, you need to modify `open_flamingo/eval/evaluate.py`. Specifically, we updated the `evaluate_vqa` function and introduced a new `ExtendedVQADataset` in `open_flamingo/eval/eval_datasets.py` to support multi-frame inputs.

For further details, please refer to the original [evaluation README](https://github.com/mlfoundations/open_flamingo/tree/main/open_flamingo/eval).

