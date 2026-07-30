from pathlib import Path
import os
import uuid
import re
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


MODEL = "openai/gpt-4.1-nano"

DB_NAME = os.path.abspath('') + "/preprocessed_db"
collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = os.path.abspath('') + "/knowledge-base"
AVERAGE_CHUNK_SIZE = 100
OVERLAP = 50
wait = wait_exponential(multiplier=1, min=10, max=240)


WORKERS = 3

openai = OpenAI()



class Document(BaseModel):
    """Represents a single raw document with content and source metadata"""
    content: str
    metadata: dict


class Chunk(BaseModel):
    """Represents a text slice optimized for vector db ingestion."""
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.content,
            "metadata": self.metadata
        }


class DocumentLoader(BaseModel):

    # Retries up to 3 times with exponential backoff if a file/network error occurs
    @retry(
        retry=retry_if_exception_type((IOError, RuntimeError)),
        stop=stop_after_attempt(3),
        wait=wait,
        reraise=True
    )

    def fetch_documents(self) -> list[Document]:
        """A homemade version of the LangChain DirectoryLoader"""

        documents = []
        dir_path = Path(KNOWLEDGE_BASE_PATH)
        md_files = list(dir_path.glob('*.md'))
        for file in md_files:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
                doc_id = str(uuid.uuid4())[:8]
                metadata = {"source": str(file), "doc_id": doc_id}
                documents.append(Document(content=text, metadata=metadata))
        return documents


class TextCleaner(BaseModel):
    """Preprocesses raw text to remove noise before chunking happens."""
    @staticmethod
    def clean(text: str) -> str:
        # Standardize spacing and strip trailing/leading whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class ChunkingStrategy(BaseModel):
    chunk_size: int
    chunk_overlap: int
    """Base class defining the contract for various chunking mechanisms."""
    def split(self, document: Document) -> list[Chunk]:
        raise NotImplementedError("Subclasses must implement the split method.")


class FixedTokenChunker(ChunkingStrategy):
    """Chunks text by words (proxy tokens) with an enforced overlapping window."""

    def split(self, document: Document) -> list[Chunk]:
        cleaned_text = TextCleaner.clean(document.content)
        words = cleaned_text.split(" ")
        chunks = []
        
        doc_id = document.metadata.get("doc_id", "unknown")
        start_idx = 0
        chunk_index = 0

        while start_idx < len(words):
            end_idx = start_idx + self.chunk_size
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            # Keep original metadata and inject chunk-specific lineage
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_index,
                "word_start": start_idx,
                "word_end": min(end_idx, len(words))
            })
            
            chunk_id = f"{doc_id}-c{chunk_index}"
            chunks.append(Chunk(chunk_id=chunk_id,
            doc_id=doc_id, content=chunk_text, metadata=chunk_metadata))
            
            # Shift the window forward by chunk size minus the overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            chunk_index += 1
            
            # Guard against infinite loops if overlap parameters are misconfigured
            if self.chunk_size <= self.chunk_overlap:
                break

        return chunks       


class IngestionPipeline:
    """Orchestrates the entire flow from raw file to database-ready chunks."""
    def __init__(self, loader: DocumentLoader, chunker: ChunkingStrategy):
        self.loader = loader
        self.chunker = chunker


    def run(self) -> list[dict[str, any]]:
        final_result = []
        try:
            documents = self.loader.fetch_documents()
            for elem in documents:
                chunks = self.chunker.split(elem)
                final_result.extend(chunks)               
            return final_result
        except Exception as e:
            print(f"Pipeline failed completely: {e}")
            return []



def create_embeddings(chunks: Chunk):
    """
    Creates embeddings and stores them in a Chroma vector database
    Args:
        chunks: chunks of text to be transformed and stored
    """

    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.content for chunk in chunks]
    emb = openai.embeddings.create(model=embedding_model, input=texts).data
    vectors = [e.embedding for e in emb]

    collection = chroma.get_or_create_collection(collection_name)

    ids = [chunk.chunk_id for chunk in chunks]
    metas = [chunk.metadata for chunk in chunks]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)
    print(f"Vectorstore created with {collection.count()} documents")


if __name__ == "__main__":
    loader = DocumentLoader()
    chunker = FixedTokenChunker(chunk_size=AVERAGE_CHUNK_SIZE, chunk_overlap=OVERLAP)
    ingestion = IngestionPipeline(loader=loader, chunker=chunker)
    chunks = ingestion.run()
    create_embeddings(chunks)
    print("Ingestion complete")
