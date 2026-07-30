import json 

def build_prompt(n: int) -> str:
    """
    This function builds a prompt for the LLM. It includes one example of JSON objects we want to get.
    Args:
        n: how many data points we would like to create
    Returns:
        system_prompt+user: the user prompt containing the system message and the number of data points
    """
    with open('helpers/example.json', 'r') as file:
        example = json.load(file)

    system_prompt = f"""
        You are a synthetic dataset generator for the casino industry.
        Generate realistic datasets.
        This is an example: {example}.
        Rules:
        1. user_name must be unique for each record
        2. the value of net_profit MUST be the result of the subtraction of total_withdrawals value from the total_deposits value
        3. the value of age MUST be higher than 17
        4. the value of either total_deposits or total_withdrawals should not be higher than 10000
        """


    user = f"""
        Create a synthetic dataset.
        DO NOT write code, comments, or explanations.

        Generate EXACTLY {n} dictionaries
        Do not create more than {n} dictionaries!!
        follow the example format EXACTLY.
        RETURN ONLY DICTIONARIES
        """

    return system_prompt + user


def get_json_list(output: str, number: int) -> list:
    """
    The goal of this function is to get a list of json objects/dictionaries.
    Because LLMs might hallucinate or give the wrong number of data points,
    we would like to clean the received text.
    Args:
        output: the output of the LLM
        number: number of data points to clip to
    Returns:
        outputlist[:number]: the sliced list containing no more than number of data points
    """
    first_index = output.find('{')
    output = output[first_index:]
    outputlist = output.split('}')
    for idx, elem in enumerate(outputlist):
        findstart = outputlist[idx].find('{')
        outputlist[idx] = outputlist[idx][findstart:] + '}'
    # check if valid
    keys = ['user_name', 'first_name', 'last_name', 'age', 'gender', 'total_deposits', 'total_withdrawals', 'net_profit']
    for elem in outputlist:
        if all(key in elem for key in keys):
            continue
        else:
            outputlist.remove(elem)
    return outputlist[:number]

def clean_raw_text(input: str, number: int) -> str:
    """
    This function aims to give a string representation of a generated list of JSON objects
    Args:
        input: raw output of the LLM
        number: number of data points to retain
    Returns:
        string_repr: string representation of a list of JSON objects for display purposes
    """
    clean_list = get_json_list(input, number)
    string_repr = str(clean_list)
    return string_repr


def save_file(input: str, number: int) -> None:
    """
    This function saves the generated list to a jsonl file
    Args:
        input: raw output of the LLM
        number: number of data points to retain
    Returns:
        None
    """
    clean_list = get_json_list(input, number)
    with open('helpers/data.jsonl', 'w', encoding='utf-8') as f:
        for item in clean_list:
            f.write(json.dumps(item) + "\n")
