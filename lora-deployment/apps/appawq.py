import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub[hf_transfer]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) # Maximize adapter download speeds
)

app = modal.App(name="vllm-awq-lora-service", image=vllm_image)
model_volume = modal.Volume.from_name("my-quantized-models")

LORA_ADAPTER_ID = "mess1989/medassistant-2026-07-26_11.24.02"

@app.cls(
    gpu="T4", # The 4-bit model runs incredibly fast on a budget-friendly A10G
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class VllmLoraServer:
    @modal.enter()
    def load_engine(self):
        from vllm import LLM
        from huggingface_hub import snapshot_download

        print("Downloading LoRA adapter weights from Hugging Face...")
        self.lora_path = snapshot_download(repo_id=LORA_ADAPTER_ID)

        print("Initializing vLLM with custom AWQ weights and LoRA engine...")
        self.llm = LLM(
            model="/models/Llama-3.2-3B-AWQ", # Path to mounted volume
            quantization="awq",             # Informs engine of 4-bit layout
            enable_lora=True,               # Activates LoRA engine architecture
            max_loras=1,                    # Number of active adapters to cache
            max_lora_rank=32,               # Must match or exceed your adapter's rank (r)
            max_model_len=512               # Set according to context limits
        )

    @modal.method()
    def generate(self, prompt: str):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling_params = SamplingParams(
            temperature=0.7, 
            max_tokens=256
        )
        
        # Instantiate request identifier for the adapter
        lora_request = LoRARequest(
            lora_name="custom_adapter", 
            lora_int_id=1, 
            lora_path=self.lora_path
        )

        print("Executing highly-parallelized inference call...")
        outputs = self.llm.generate(
            prompt, 
            sampling_params=sampling_params, 
            lora_request=lora_request
        )
        return outputs[0].outputs[0].text

    # Web endpoint for production REST API integrations
    @modal.fastapi_endpoint(method="POST")
    def api_generate(self, item: dict):
        prompt = item.get("prompt", "")
        response = self.generate.local(prompt)
        return {"response": response}
