import argparse
import importlib
from PIL import Image
from open_flamingo.train.distributed import init_distributed_device, world_info_from_env

# Constants
MODEL_NAME = "open_flamingo"
CHECKPOINT_PATH = "/home/anhnv16/maund/open-flamingo-3D/OpenFlamingo-3B-vitl-mpt1b/checkpoint_19.pt"


# Model generation parameters
GENERATION_CONFIG = {
    "min_generation_length": 2,
    "max_generation_length": 5,
    "num_beams": 3,
    "length_penalty": 0,
}

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        help="Model name. Currently only `OpenFlamingo` is supported.",
        default=MODEL_NAME,
    )
    parser.add_argument("--results_file", type=str, default=None, help="JSON file to save results")

    # Distributed training arguments
    parser.add_argument("--dist-url", default="env://", type=str, help="URL for distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="Distributed backend")
    parser.add_argument("--horovod", action="store_true", help="Use Horovod for distributed training")
    parser.add_argument(
        "--no-set-device-rank",
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES is restricted to one per process).",
    )

    args, _ = parser.parse_known_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    
    return args

def setup_model(device_id):
    """Loads the model dynamically and initializes it."""
    module = importlib.import_module(f"open_flamingo.eval.models.{MODEL_NAME}")

    model_args = {
        "device": -1,
        "vision_encoder_path": "ViT-L-14",
        "vision_encoder_pretrained": "openai",
        "lm_path": "anas-awadalla/mpt-1b-redpajama-200b",
        "lm_tokenizer_path": "anas-awadalla/mpt-1b-redpajama-200b",
        "cross_attn_every_n_layers": 1,
        "checkpoint_path": CHECKPOINT_PATH,
        "precision": "amp_bf16",
    }

    model = module.EvalModel(model_args)
    model.set_device(device_id)
    model.init_distributed()
    
    return model

def load_images(img_paths):
    """Loads images and converts them to RGB format."""
    return [[Image.open(img_path).convert("RGB") for img_path in img_paths]]

def main():
    """Main execution function."""
    args = parse_arguments()
    device_id = init_distributed_device(args)

    # Load and set up model
    model = setup_model(device_id)

    

    # Define prompt
    PROMPT = ["<image><image><image>What does this illustration show?"]
    
    FRAME_PATHS = [
        "/maund/dataset/textvqa/train_images/dfe5e9ddcf2f1da6.jpg",
        "/maund/dataset/textvqa/train_images/a2dcb8b3f96e0fb2.jpg",
        "/maund/dataset/textvqa/train_images/673ca66b35af14d7.jpg",
    ]
    # Load images
    batch_3D_frames = load_images(FRAME_PATHS)

    # Generate outputs
    outputs = model.get_outputs(batch_images=batch_3D_frames, batch_text=PROMPT, **GENERATION_CONFIG)

    # Print output
    print("Output:", outputs)

if __name__ == "__main__":
    main()
