#globals_space.py
"""
GLOBAL SETTINGS / SHARED STATE

This file stores:
- Configuration constants (paths, model selections, context amount)
- Shared runtime objects (vector store, retriever)
- Logging configuration

Multiple modules (main/tools/rag) need access to the same objects:
- the vector store
- the retriever
- model settings
"""

import pathlib
import langchain_ollama
import logging
from pandasai_litellm.litellm import LiteLLM

# Runtime objects, will be initialized at runtime
_VECTOR_STORE = None    # Will hold the vector store object
_RETRIEVER = None       # Will hold the retriever object from the vector store

# Location of the data folder, change this to your data path if needed
_DATA_FOLDER = pathlib.Path(r"./Data")

# Dataset metadata dictionary used by the auto_analyse tool
_DATASET_METADATA = {}

# Logger setup
logger = logging.getLogger(__name__)
def configure_logger() -> None:
    """
    Configure the logger for the whole project.
    Logs go to the terminal, which is useful for debugging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Settings definitions

# Agent model (chat model) that decides whether to call tools and generates final answers
_AGENT_MODEL = langchain_ollama.ChatOllama(
    model="gpt-oss:20b",
    validate_model=True,    # Checks if the model exists in Ollama
    temperature=0.3         # Lower temperature = more deterministic responses
)

# PandasAI uses an LLM to generate code/queries to answer questions about a DataFrame.
_AUTO_ANALYSE_MODEL = LiteLLM(
    model = "ollama/gpt-oss:20b",
    api_base = "http://localhost:11434"
)

# Embedding model (creates vectors from text) used for the vector database (RAG)
_EMBEDDING_MODEL = langchain_ollama.OllamaEmbeddings(model="mxbai-embed-large:335m")

# Folder where Chroma stores the vector store
_VECTOR_STORE_PATH = "./chroma_location_embeddings"



# Amount of context chunks the agent will retrieve from vector database
_CONTEXT_AMOUNT = 5