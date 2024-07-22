from transformers import AutoModelForCausalLM, AutoTokenizer

# Model checkpoints
models_checkpoints = [
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "HuggingFaceTB/SmolLM-1.7B",
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B"
]

# Dictionary to store models and tokenizers
models_and_tokenizers = {}

# Loading models and tokenizers
for checkpoint in models_checkpoints:
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    models_and_tokenizers[checkpoint] = (model, tokenizer)

