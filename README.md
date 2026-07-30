## Summary
The following repository contains the code for projects created under the umbrella of Ed Donner's AI Agentic Core course, which I finished in July 2026

## Subdirectories
1. The **small-ideas** subdirectory contains code snippets for the small toy projects
- **api-translator-wo-rag** contains a comparison of cutoff points for two models. The aim of the tool is to migrate a piece of code using Sagemaker SDK V2 into into Sagemaker SDK V3. Older models are unable to do that given how recent the API change is. Newer models are perfectly capable, but their cost is much higher.
- **gradio-primer** contains a simple web app built with the use of Gradio. It is supposed to answer questions about a popular data science library called pandas.
- **generator** contains an LLM-using-only generator for synthetic data. It does NOT contain any structured outputs defined through pydantic; it only exists to see the limit of an LLM if not given any other constraints or requirements.
2. The **api-translator-with-rag** contains all the code required to run a tool that explains and helps with migration from Sagemaker SDK V2 into into Sagemaker SDK V3. Older models are unable to do that given how recent the API change is, however, the subdirectory also contains code for crawling the sagemaker docs, scraping the relevant notebooks, and turning them into .MD files.
3. The **lora-deployment** contains full code related to fine-tuning a base model; it links to a relevant fine-tuning dataset, pushing LoRA adapters to HuggingFace spaces, and then downloading them back to deploy them together with a quantized base model on Modal. Several ways of quantizing the base model are shown. 
