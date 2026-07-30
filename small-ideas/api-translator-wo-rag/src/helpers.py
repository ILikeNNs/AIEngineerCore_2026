system_prompt = f"""
Your task is to convert code containing the SageMaker Python SDK V2 into code containing the SageMaker Python SDK V3.
Respond only with code containing the SageMaker Python SDK V3. Do not provide any explanation other than occasional comments.
"""

def user_prompt_for(python: str) -> str:
    """
    This function contains the user prompt
    Args:
        python: contains the python code snippet with V2 version of Sagemaker SDK
    Returns:
        an f-string with instructions and pasted python code
    """
    return f"""Port this SageMaker Python SDK V2 code into SageMaker Python SDK V3 code 
    with the implementation that produces identical output.
    Respond only with code. Code to port:
    ```python {python}```
    """

def messages_for(python: str) -> list:
    """
    This function compiles the system and the user prompts
    Args:
        python: contains the python code snippet with V2 version of Sagemaker SDK
    Returns:
        a list with system and user messages
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(python)}
    ]