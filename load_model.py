import transformers
import torch
import time
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face token (HF_TOKEN) not found in environment variables.")

import os

login(token=hf_token)

# ---------------------------
# CONFIG
# ---------------------------
MODEL_PATH = "Sunbird/Sunflower-14B"
SYSTEM_MESSAGE = (
    "You are Sunflower, a multilingual assistant made by Sunbird AI who understands all "
    "Ugandan languages. You specialise in accurate translations, explanations, summaries and other cross-lingual tasks."
)
RETRY_COUNT = 2
RETRY_DELAY_SECONDS = 5

# ---------------------------
# DEVICE SELECTION
# ---------------------------
def choose_device():
    """
    Select the best available device:
    - CUDA (NVIDIA GPU)
    - MPS (Apple Silicon)
    - CPU (fallback)
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # MPS prefers float32 for stability, but bfloat16 works experimentally on macOS 14+
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float32

DEVICE, TORCH_DTYPE = choose_device()
print(f"[INFO] Using device: {DEVICE}, dtype: {TORCH_DTYPE}")

# ---------------------------
# MODEL & TOKENIZER LOAD (once)
# ---------------------------
def load_sunbird_model(model_path=MODEL_PATH, torch_dtype=TORCH_DTYPE, device=DEVICE):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    try:
        if device.type == "cuda":
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype,
                device_map="auto",
            )
        else:
            # MPS or CPU — load to RAM then move manually
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
    except Exception as e:
        print(f"[Sunbird] Device load failed ({device}), falling back to CPU: {e}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )
        device = torch.device("cpu")
    return tokenizer, model, device

# Load model/tokenizer
try:
    tokenizer, sunbird_model, DEVICE = load_sunbird_model()
    sunbird_model.eval()
    print(f"[Sunbird] Model loaded successfully on {DEVICE}.")
except Exception as e:
    raise RuntimeError(f"Failed to load Sunbird model at {MODEL_PATH}: {e}")