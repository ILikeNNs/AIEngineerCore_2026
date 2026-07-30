## The introduction
The following subdirectory contains code necessary to run and deploy a fine-tuning model. The use case is flashcards for medical students, and the base model is Llama-3.2-3B.

## Project structure
The project contains several notebooks in the root directory:
- **EDA.ipynb** contains a very brief analysis of the provided dataset; missing data points are removed, and the distribution of words/tokens to define a maximum input token size is established.
- **medicalfinetuning.ipynb** contains the process of finetuning a base model. Google Colab's T4 instance was used for that. The base model was first quantized with BitsAndBytes to 4bit precision. LoRA adapters were pushed to HuggingFace spaces.
- **checknotebook.ipynb** contains the results of prompting the fine-tuned model.

There are also several directories that are necessary for the deployment on Modal:
- **quantizers** contains two ways of quantizing the model directly on Modal
    - **quantizeawq** includes code leveraging the deprecated awq library (it is no longer maintained)
    - **quantizecompressor** includes code leveraging the llmcompressor library maintained by vllm library creators
- **apps** contains several ways of deploying the model on Modal:
    - **app.py** includes no quantization (non-quantized based model + LoRA adapters)
    - **appawq.py** includes deployment of a model quantized with awq
    - **appcompressor.py** includes deployment of a model quantized with llmcompressor
    - **apptrpeft.py** includes deployment of a model quantized with bitsandbytes (like in the notebooks)
- **APItests** contains two ways of testing whether a deployed endpoint is processing requests correctly
    - **test.py** includes code for any of the above methods of quantization except the one with bitsandbytes (different request format)
    - **testtrpeft.py** same but for transformers+peft (bitsandbytes quantization)


## TO-DO
1. Refactor the notebooks to make greater use of separate 'utils' python scripts
2. Remove unnecessary and repeated code - modularize further
3. Recheck whether all functions contain docstrings with Args and Returns sections