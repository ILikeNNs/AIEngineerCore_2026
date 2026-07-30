import modal

quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "llmcompressor>=0.11.0",  # Modern vLLM optimization engine
        "transformers>=4.48.0",
        "datasets",
        "accelerate",
        "torch"
    )
)

app = modal.App(name="llm-compressor-quantizer", image=quant_image)
model_volume = modal.Volume.from_name("my-quantized-models", create_if_missing=True)

MODEL_ID = "meta-llama/Llama-3.2-3B" 
QUANT_SAVE_PATH = "/models/Llama-3.2-3B-LLMc"

@app.function(
    gpu="T4",  # Requires strong GPU to load unquantized weights + process dataset
    volumes={"/models": model_volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def run_quantization():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    print(f"Loading full-precision base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )

    print("Loading calibration dataset...")
    # Load sample data required by AWQ to capture activation spikes
    dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
    
    def preprocess(example):
        return {"text": example["instruction"] + " " + example["output"]}
    
    # Format and tokenize data samples
    dataset = dataset.map(preprocess)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=512)
    calib_dataset = dataset.map(tokenize_function, batched=True)

    # Reconstruct the recipe using the new modular layout
    recipe = [
        AWQModifier(),
        QuantizationModifier(
            targets="Linear", 
            scheme="W4A16_ASYM", # 4-bit weights, 16-bit activation tensors
            ignore=["lm_head"]   # Protect language head output precision
        )
    ]

    print("Executing One-Shot Compression Pipeline...")
    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=128
    )

    print(f"Saving modern compressed-tensor weights to: {QUANT_SAVE_PATH}")
    model.save_pretrained(QUANT_SAVE_PATH)
    tokenizer.save_pretrained(QUANT_SAVE_PATH)
    
    model_volume.commit()
    print("Optimization successful. Cloud Volume Committed!")
