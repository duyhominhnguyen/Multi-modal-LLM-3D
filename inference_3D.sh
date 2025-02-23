
DATASET="VLM_ADNI_DATA"
NOTE=original_LLaVA-Med_new
NOTE_OUTPUT="_pre-train_stage_2_3D_mlp" # lora: with adapter again, non: no adapter.
DATASET_LINK="/netscratch/duynguyen/Research/bao_llava_med/Dense/dataset_3D/$DATASET"

#VISION_TOWER=goog§le/siglip-so400m-patch14-384
VISION_TOWER=openai/clip-vit-large-patch14

EPOCH=2
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
STEP=4

python3 llava/eval/model_vqa.py \
	--model-path weights/llava_$DATASET-$EPOCH-epo$NOTE$NOTE_OUTPUT \
	--question-file $DATASET_LINK/AD_caption-llava_3D_version.json  \
	--image-folder $DATASET_LINK/vbm_images \
	--answers-file results/${DATASET}$NOTE.jsonl \
	--conv-mode llava_llama_2 \
	--temperature 0.1 