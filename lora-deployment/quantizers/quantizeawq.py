import modal

quant_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "autoawq>=0.2.6",
        "transformers",
        "accelerate",
        "torch"
    )
)

app = modal.App(name="awq-quantizer", image=quant_image)

# Persistent volume to store the 4-bit weights across apps
model_volume = modal.Volume.from_name("my-quantized-models", create_if_missing=True)

MODEL_ID = "meta-llama/Llama-3.2-3B" # The unquantized FP16 model
QUANT_SAVE_PATH = "/models/Llama-3.2-3B-AWQ"

@app.function(
    gpu="T4",  # Requires an A100 to fit the full-precision unquantized model
    volumes={"/models": model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")] # If using gated models
)
def run_quantization():
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"Loading full-precision base model: {MODEL_ID}")
    model = AutoAWQForCausalLM.from_pretrained(
        MODEL_ID, 
        low_cpu_mem_usage=True, 
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 4-bit GEMM configuration for optimal vLLM performance
    quant_config = {
        "zero_point": True, 
        "q_group_size": 128, 
        "w_bit": 4, 
        "version": "GEMM"
    }

    print("Running AWQ calibration (this takes a few minutes)...")
    model.quantize(tokenizer, quant_config=quant_config)

    print(f"Saving 4-bit weights to storage volume: {QUANT_SAVE_PATH}")
    model.save_quantized(QUANT_SAVE_PATH)
    tokenizer.save_pretrained(QUANT_SAVE_PATH)
    
    # Save files to cloud storage
    model_volume.commit()
    print("Quantization complete and saved!")
