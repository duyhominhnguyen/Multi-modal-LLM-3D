torchrun --nnodes=1 --nproc_per_node=1 open_flamingo/train/train.py \
  --lm_path anas-awadalla/mpt-1b-redpajama-200b \
  --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
  --cross_attn_every_n_layers 1 \
  --dataset_resampled \
  --batch_size_mmc4 2 \
  --train_num_samples_mmc4 10\
  --workers=4 \
  --run_name OpenFlamingo-3B-vitl-mpt1b \
  --num_epochs 2 \
  --warmup_steps  1875 \
  --mmc4_textsim_threshold 0.24 \
  --mmc4_shards "modifications/VLM_ADNI_DATA/replicate_mmc4/{000000000..000000001}.tar" \


# --laion_shards "modifications/laion-high-resolution/{00000..00001}.tar" \
# --mmc4_shards "modifications/mmc4_shards/{000000000..000000001}.tar" \
# --train_num_samples_laion 20 \
#   --batch_size_laion 4 \
#   --loss_multiplier_laion 0.2 \






# torchrun --nnodes=1 --nproc_per_node=4 open_flamingo/train/train.py \
#   --lm_path anas-awadalla/mpt-1b-redpajama-200b \
#   --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
#   --cross_attn_every_n_layers 1 \
#   --dataset_resampled \
#   --batch_size_mmc4 32 \
#   --batch_size_laion 64 \
#   --train_num_samples_mmc4 125000\
#   --train_num_samples_laion 250000 \
#   --loss_multiplier_laion 0.2 \
#   --workers=4 \
#   --run_name OpenFlamingo-3B-vitl-mpt1b \
#   --num_epochs 480 \
#   --warmup_steps  1875 \
#   --mmc4_textsim_threshold 0.24 \
#   --laion_shards "/path/to/shards/shard-{0000..0999}.tar" \
#   --mmc4_shards "/path/to/shards/shard-{0000..0999}.tar" \
#   --report_to_wandb