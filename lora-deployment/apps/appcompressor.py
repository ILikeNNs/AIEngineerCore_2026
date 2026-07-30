import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub[hf_transfer]"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(name="vllm-compressor-inference", image=vllm_image)
model_volume = modal.Volume.from_name("my-quantized-models")

# Replace with your adapter repository info
LORA_ADAPTER_ID = "mess1989/medassistant-2026-07-26_11.24.02"

@app.cls(
    gpu="T4", # Compressed 4-bit model runs with excellent VRAM economy
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class VllmCompressorServer:
    @modal.enter()
    def load_engine(self):
        from vllm import LLM
        from huggingface_hub import snapshot_download

        print("Caching LoRA adapter weights...")
        self.lora_path = snapshot_download(repo_id=LORA_ADAPTER_ID)

        print("Initializing vLLM Engine...")
        # Note: 'quantization' argument is omitted. vLLM auto-detects 
        # the 'compressed-tensors' format inside the folder configurations.
        self.llm = LLM(
            model="/models/Llama-3.2-3B-LLMc",
            enable_lora=True,          
            max_loras=1,               
            max_lora_rank=32,          
            max_model_len=512         
        )

    @modal.method()
    def generate(self, prompt: str):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
        lora_request = LoRARequest("my_adapter", 1, self.lora_path)

        outputs = self.llm.generate(prompt, sampling_params, lora_request=lora_request)
        return outputs[0].outputs[0].text

    @modal.web_endpoint(method="POST")
    def api_generate(self, item: dict):
        prompt = item.get("prompt", "")
        response = self.generate.local(prompt)
        return {"response": response}
