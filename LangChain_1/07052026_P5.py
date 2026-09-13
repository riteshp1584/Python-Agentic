import os
import torch
from dotenv import load_dotenv
from transformers import pipeline
import warnings

# 1. Block Hugging Face and Python warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")

load_dotenv()

# 1. Cleaner Pipeline Setup
# Moving dtype and cache_dir into model_kwargs silences the deprecation warnings
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "cache_dir": os.getenv("HF_HOME")
    },
    device_map="auto"
)

# 2. Silencing the Tokenizer warning
pipe.tokenizer.clean_up_tokenization_spaces = False

# 3. Requesting the Quant Logic
messages = [
    {"role": "system", "content": "You are a concise Quant Developer."},
    {"role": "user", "content": "Write Python code to calculate a 20-day MA on the 'close' column of a dataframe."}
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Increased max_new_tokens to 1000 to prevent cut-offs
# Removed any conflicting parameters to keep the console clean
outputs = pipe(
    prompt,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.1 # Low temperature for reliable code generation
)

print("\n--- OUTPUT ---")
print(outputs[0]["generated_text"].split("<|assistant|>")[-1])
