import modal
import os

# Define the container image with essential quantization libraries
bnb_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.10.0",
        "torch>=2.2.0",
        "fastapi"
    )
)

app = modal.App(name="bnb-lora-inference", image=bnb_image)

# Constants for your specific configuration
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"  # The unquantized base model
LORA_ADAPTER_ID = "mess1989/medassistant-2026-07-26_11.24.02" # Your adapter paths

@app.cls(
    gpu="T4", # 8B model in 4-bit easily fits inside A10G's 24GB VRAM
    secrets=[modal.Secret.from_name("huggingface-secret")], # For gated models/private adapters
    concurrency_limit=5 # Helps manage peak GPU scaling
)
class BnbLoraModel:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        print("Initializing 4-bit BitsAndBytes Configuration...")
        # Configure NF4 (NormalFloat4) quantization settings
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",         # Recommended type for optimal accuracy
            bnb_4bit_use_double_quant=True,    # Quantizes quantization constants for extra VRAM savings
            bnb_4bit_compute_dtype=torch.bfloat16 # Speeds up execution; use torch.float16 if on older GPUs
        )

        print(f"Loading base model into 4-bit memory: {BASE_MODEL_ID}")
        # Base model is downloaded and compressed directly into GPU memory
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto" # Automatically handles layer placement
        )

        print(f"Loading and applying LoRA adapter: {LORA_ADAPTER_ID}")
        # Dynamically attach the adapter weights on top of the 4-bit base model
        self.model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        # Enforce padding token configuration if not natively set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @modal.method()
    def generate(self, prompt: str):
        import torch

        # Tokenize inputs and map them directly to the active GPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        print("Generating text with active LoRA layers...")
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Decode and slice out the original prompt from final text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text[len(prompt):]

    # Exposes your container logic over HTTPS as a secure webhook endpoint
    @modal.web_endpoint(method="POST")
    def api_generate(self, item: dict):
        prompt = item.get("prompt", "")
        response = self.generate.local(prompt)
        return {"response": response}
