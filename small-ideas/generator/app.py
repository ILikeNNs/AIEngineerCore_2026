import os
import re
from dotenv import load_dotenv
from helpers.helpers import build_prompt, get_json_list, clean_raw_text, save_file
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig
)
from IPython.display import Markdown, display
import gradio as gr
import torch

# loading my local env and the API key
load_dotenv(override=True)
hf_key = os.getenv('HF_KEY')

# model I preinstalled
MODEL = "microsoft/Phi-3-mini-4k-instruct"


if torch.cuda.is_available():
    torch.cuda.empty_cache()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    quantization_config=quant_config,
    local_files_only=True,
    low_cpu_mem_usage=True 
)


def generate_dataset(number):
    prompt = build_prompt(number)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_tokens = output[0][inputs.input_ids.shape[-1]:]

    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    text = text.replace("\n", "")
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def main():
    """
    The aim of this function is to run the Gradio UI app.
    """
    with gr.Blocks(title="Synthetic Data Generator") as demo:

        gr.Markdown(
            """
            # Synthetic Data Generator

            Generate realistic synthetic datasets instantly using AI.
            """
        )

        outputraw = gr.Textbox(
            label="raw data",
            placeholder="dictionaries which will be put in a jsonl file",
            lines=8
        )

        output = gr.Textbox(
            label="data",
            placeholder="dictionaries which will be put in a jsonl file",
            lines=8
        )

        rows = gr.Slider(
            minimum=1,
            maximum=5,
            value=3,
            step=1,
            label="Number of Examples"
        )

        generate_btn = gr.Button(
            "Generate raw text",
            variant="primary"
        )

        clean_btn = gr.Button(
            "Clean text"
        )

        save_btn = gr.Button(
            "Save the file"
        )


        generate_btn.click(
            fn=generate_dataset,
            inputs=[rows],
            outputs=[outputraw]
        )

        clean_btn.click(
            fn=clean_raw_text,
            inputs=[outputraw, rows],
            outputs=[output]
        )

        save_btn.click(
            fn=save_file,
            inputs=[outputraw, rows]
        )

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
