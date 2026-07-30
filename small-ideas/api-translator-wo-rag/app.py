# imports
import os
from dotenv import load_dotenv
from openai import OpenAI
from src.code_snippet import pythoncode
from src.helpers import user_prompt_for, messages_for
import gradio as gr
from IPython.display import Markdown, display

# loading my local env and the API key
load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI(api_key=openai_api_key)

# mini cutoff was a few years ago, luna's in the first half of the current year: 2026
models = ['gpt-4.1-mini', 'gpt-5.6-luna']


def translate(snippet: str, model: str) -> str:
    """
    The aim of the function is to call the LLM and make a translation from V2 to V3
    Args:
        snippet: the code snippet to translate
        model: the model to choose (old or new)
    Returns:
        reply: the response of the LLM
    """
    response = openai.chat.completions.create(model=model, messages=messages_for(snippet))
    reply = response.choices[0].message.content
    return reply


def main():
    """
    The aim of this function is to run the Gradio UI app.
    """
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks() as ui:
        with gr.Row():
            v2 = gr.Textbox(label="V2 code:", lines=28, value=pythoncode)
            v3 = gr.Textbox(label="V3 code:", lines=28)
        with gr.Row():
            model = gr.Dropdown(models, label="Select model", value=models[0])
            convert = gr.Button("Convert code")

        convert.click(translate, inputs=[v2, model], outputs=[v3])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
