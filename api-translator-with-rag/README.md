# APItranslator (Sagemaker SDK V2 into Sagemaker SDK V3)

The aim of this project is to give answers about migrating from Sagemaker SDK v2 to Sagemaker SDK v3.<br>
I want to do this with an old and cheap OpenAI gpt-4.1-nano model, but its cutoff point precedes the release of the V3 SDK.<br>
To go around the cutoff point, I want to build a vector database containing examples derived from Jupyter notebooks.<br>
The Jupyter notebooks are created by the Sagemaker creators, and available online for free.<br>

There are several modules for the full pipeline:
- `crawler.py` contains a custom crawler that searches the sagemaker.readthedocs.io website and picks up V3 doc examples (only .ipynb files)
- `datagenerator.py` contains code to download the .ipynb files, and turn them into Markdown ones (ipynb format cannot be stored directly into Chroma)
- `ingest.py` takes the Markdown files and ingests them into a vector database (Chroma)
- `answer.py` contains the LLM code. It includes functions for calling an LLM to rewrite the user's query, as well as a custom reranking function.

The code can be run by (when using uv):
> uv run app.py


### NOTE:
If you use the code to download ipynb files you might hit a limit and bounce off the file download because of a Cloudflare block. Retry later in such a case.