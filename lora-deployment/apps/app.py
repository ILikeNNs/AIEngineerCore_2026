import os
import modal

# 1. Define your Modal App and Volume cache
app = modal.App("vllm-lora-service")
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Define IDs (Replace these with your exact repo names)
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B" 
LORA_ADAPTER_ID = "mess1989/medassistant-2026-07-26_11.24.02"

# 2. Build the container environment
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}) # Fast Rust-based downloads
)

# 3. Build the serverless Class
@app.cls(
    image=vllm_image,
    gpu="T4",  # Adjust GPU size depending on the base model (e.g., L4, A100, H100)
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")], # Needed if model or adapter is gated/private
    scaledown_window=300,
)
class VLLMLoraServer:
    @modal.enter()
    def load_models(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from huggingface_hub import snapshot_download

        # Pre-download the adapter into the cache volume
        print("Ensuring LoRA adapter is cached...")
        self.lora_path = snapshot_download(repo_id=LORA_ADAPTER_ID)

        # Initialize the vLLM Engine with LoRA enabled
        print("Initializing vLLM Engine...")
        engine_args = AsyncEngineArgs(
            model=BASE_MODEL_ID,
            enable_lora=True,
            max_lora_rank=32, # Ensure this matches or exceeds your trained LoRA rank
            max_model_len=4096
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def generate(self, prompt: str):
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
        
        # params: (human_readable_name, unique_int_id, local_path_to_adapter)
        lora_request = LoRARequest("my_adapter", 1, self.lora_path)

        # Submit request to engine
        results_generator = self.engine.generate(
            prompt, 
            sampling_params, 
            request_id=f"req-{os.urandom(4).hex()}",
            lora_request=lora_request
        )

        # Aggregate tokens
        final_output = ""
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text
            
        return final_output

# 4. Optional: Expose it as a standard Web endpoint
@app.function(image=vllm_image)
@modal.fastapi_endpoint(method="POST")
async def api(prompt: str):
    server = VLLMLoraServer()
    return {"text": await server.generate.remote.aio(prompt)}
