import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
# MODEL = "groq/openai/gpt-oss-120b"
DB_NAME = os.path.abspath('') + "/preprocessed_db"
KNOWLEDGE_BASE_PATH = os.path.abspath('') + "/knowledge-base"
SUMMARIES_PATH = os.path.abspath('') + "/summaries"


collection_name = "docs"
embedding_model = "text-embedding-3-large"
wait = wait_exponential(multiplier=1, min=10, max=240)

openai = OpenAI()

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(collection_name)

RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = """You are an expert technical assistant specializing in migrating code from AWS SageMaker Python SDK V2 to SDK V3. 
Your task is to answer the user's migration query by synthesizing your internal knowledge with the provided RAG context.
If the user asks about ANYTHING unrelated to Sagemaker, say you only answer questions about Sagemaker.
You must strictly execute your reasoning sequentially through the requested JSON schema fields.
"""

# we want to make a structured input where the user gets info from both
# the model's internal knowledge and the rag context
USER_TEMPLATE = """### USER QUERY
{user_query}

### PROVIDED RAG CONTEXT
{rag_context}"""


class Document(BaseModel):
    """Represents a single Markdown document"""
    content: str
    metadata: dict


class RankOrder(BaseModel):
    """Represents a ranker that orders the relevance of chunks"""
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


class MigrationReasoningChain(BaseModel):
    """Represents a reasoning chain that analyzes general knowledge and context, and gives a final answer separately"""
    general_knowledge: str = Field(
        description="A summary of general background information, core concepts, and coding examples of SageMaker SDK V2 related to the query. Compiled before reviewing context."
    )
    context_analysis: str = Field(
        description="Analysis of the provided RAG context fragments, identifying key facts that relate directly to the SDK V2 knowledge."
    )
    final_answer: str = Field(
        description="A definitive, cohesive synthesis translating the legacy V2 concepts into the correct SageMaker SDK V3 solution."
    )


@retry(wait=wait)
def rerank(question: str, chunks: list) -> list:
    """Call an LLM to rank the chunks in decreasing importance order.
    Args:
        question: a question provided by the user
        chunks: list of chunks picked from the vector database
    Returns:
        list: a list of reordered chunks
     """

    system_prompt = """
    You are a document re-ranker. 
    Rank the provided text chunks based on their relevance to the user's question, placing the most relevant chunk first.
    Output only a valid JSON array containing all provided chunk IDs in their new order. 
    Do not include any introductory or concluding text.
    """

    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def make_rag_messages(question: str, history: list, chunks: list) -> list:
    """
    Create messages in a predefined format with the user question and the rag context
    Args:
        question: the question posed by the user
        history: the history of the conversation so far
        chunks: a list of provided chunks
    Returns:
        list: a list containing the system message, the conversation history, and the user prompt 
    """

    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.content}" for chunk in chunks
    )

    user_prompt = USER_TEMPLATE.format(user_query=question, rag_context=context)
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_prompt}]
    )

@retry(wait=wait)
def rewrite_query(question: str, history: list = []) -> str:
    """Rewrite the user's question to be a more specific question that is more likely 
    to surface relevant content in the Knowledge Base.
    Args:
        question: the question posed by the user
        history: the history of the conversation
    Returns:
        str: the user question rephrased by the LLM
    """
    
    message = f"""
    You are in a conversation with a user, answering questions about migrating from Sagemaker SDK V2 to Sagemaker SDK V3.
    You are about to look up information in a Knowledge Base to answer the user's question.
    This is the history of your conversation so far with the user:
    {history}
    And this is the user's current question:
    {question}
    Respond only with a short, refined question that you will use to search the Knowledge Base.
    It should be a VERY short specific question most likely to surface content. Focus on the question details.
    IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
    """

    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks: list, reranked: list) -> list:
    """
    Merge chunks after rewriting the question
    Args:
        chunks: a list of chunks in original order
        reranked: a list of chunks in reviewed order
    Returns:
        list: a list containing chunks from both input lists
    """

    merged = chunks[:]
    existing = [chunk.content for chunk in chunks]
    for chunk in reranked:
        if chunk.content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question: str) -> list:
    """
    Fetch chunks without reranking
    Args:
        question: user prompt
    Returns:
        list: a list containing unranked chunks
    """

    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Document(content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question: str) -> list:
    """
    Fetch chunks with reranking and after rewriting the query
    Args:
        original_question: original user prompt
    Returns:
        list: a list containing ranked chunks
    """
    # we try to use an LLM to rewrite a question we had to make it better
    print('rewrite a question')
    rewritten_question = rewrite_query(original_question)
    # fetching unranked chunks for both original and rewritten question
    print('fetch chunks')
    chunks1 = fetch_context_unranked(original_question)
    chunks2 = fetch_context_unranked(rewritten_question)
    # merging chunks from both questions
    print('merge chunks')
    chunks = merge_chunks(chunks1, chunks2)
    # we call an LLM to help us rerank the chunks
    print('reranking')
    reranked = rerank(original_question, chunks)
    # we take only top K
    print(len(reranked))
    return reranked[:FINAL_K]


# @retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, str, str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    Args:
        question: user prompt
        history: the history of the conversation
    Returns:
        tuple[str, str, str, list]: a tuple containing three strings (structured output) and relevant chunks 
    """
    # we first get chunks we need for RAG context
    print('retrieve chunks')
    chunks = fetch_context(question)
    # we make the messages that follow our user template
    print('make rag mess')
    messages = make_rag_messages(question, history, chunks)
    # we expect a response that follows the predefined format
    print('await a response')
    response = openai.beta.chat.completions.parse(model=MODEL,
    temperature=0.2,
    messages=messages,
    response_format=MigrationReasoningChain)
    # response = completion(model=MODEL, messages=messages)
    # the output will have three parts
    # print(response)
    output = response.choices[0].message.parsed
    print('output generated')
    return output.general_knowledge, output.context_analysis, output.final_answer, chunks