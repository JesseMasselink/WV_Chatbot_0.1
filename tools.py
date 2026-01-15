# tools.py
"""
TOOLS MODULE

This file contains functions that the agent can call.

Two tool categories exist:
1) RAG Retrieval Tool: searches the vector store for relevant context.
2) Auto Analyse Tool: selects a CSV and uses PandasAI to compute an answer.

Important:
- The @tool wrappers are required by LangChain so the agent can call these functions.
- Under the hood, these tools use globals_space._RETRIEVER and globals_space._DATASET_METADATA.
"""
# Standard libraries
import os

import pandas as pd

import pandasai as pai
from langchain.tools import tool
from langchain_chroma import Chroma

from rag import load_all_csvs, build_summary_chunks, build_chroma_vector_store, get_retriever
import globals_space
from pathlib import Path

# PandasAI configuration with the selected LLM in globals_space
pai.config.set({
    "llm": globals_space._AUTO_ANALYSE_MODEL,
    "verbose": True,
})

# Function to build the vector store
def check_vector_store():
    """Check if vector store is already built."""
    # If the vector store folder exists, load it (fast startup).
    # Otherwise, try to build it (slower because embeddings must be generated).
    # If both fail, continue without RAG capabilities.
    if os.path.exists(globals_space._VECTOR_STORE_PATH):
        try:
            # Initialise _VECTOR_STORE with globals_space settings
            globals_space._VECTOR_STORE = Chroma(
                persist_directory=globals_space._VECTOR_STORE_PATH, 
                embedding_function=globals_space._EMBEDDING_MODEL,
                collection_name="location_summaries")
            # Check if vector store is functional
            globals_space._RETRIEVER = get_retriever(globals_space._VECTOR_STORE, globals_space._CONTEXT_AMOUNT)
            globals_space.logger.info("Vectorstore is succesfully located!")
        except Exception as e:
            globals_space.logger.warning(f"Could not load vector store: {e}. For now, continuing without retrieval function.")
            globals_space._VECTOR_STORE = None
            globals_space._RETRIEVER = None
    else:
        try:
            globals_space.logger.info(f"Vector store not found at {globals_space._VECTOR_STORE_PATH}. Building new vector store...")
            globals_space._VECTOR_STORE = build_vector_store()
            globals_space._RETRIEVER = get_retriever(globals_space._VECTOR_STORE, globals_space._CONTEXT_AMOUNT)
            globals_space.logger.info("Vectorstore is succesfully built!")
        except Exception as e:
            globals_space.logger.warning(f"Could not build vector store: {e}. For now, continuing without retrieval function.")
            globals_space._VECTOR_STORE = None
            globals_space._RETRIEVER = None

def build_vector_store():
    """Build the vector store from the waste management CSV dataset."""
    # Load and merge all CSV data into one Dataframe
    globals_space.logger.info("Build_vector_store function called.\n")

    globals_space.logger.info("Pre-processing: Loading waste management data...")
    df = load_all_csvs(globals_space._DATA_FOLDER)
    
    # Convert each container row into a readable text chunk
    globals_space.logger.info("Building summary chunks...")
    chunks = build_summary_chunks(df)

    # Build and persist the vector store from those chunks
    globals_space.logger.info("Building vector store...")
    globals_space._VECTOR_STORE = build_chroma_vector_store(chunks, globals_space._EMBEDDING_MODEL, globals_space._VECTOR_STORE_PATH)

    # Create a retriever so its possible to ask: "find top-k relevant chunks for this query"
    globals_space._RETRIEVER = globals_space._VECTOR_STORE.as_retriever(search_kwargs={"k": globals_space._CONTEXT_AMOUNT})

    return globals_space._VECTOR_STORE

# Metadata building for dataset selection
def build_dataset_metadata():
    """Populate the DATASET_METADATA dict by scanning all available CSV files"""
    globals_space.logger.info("Build_dataset_metadata function called!\n")

    folder = Path(globals_space._DATA_FOLDER)
    if not folder.exists():
        globals_space.logger.warning(f"Data folder not found: {folder}")
        return

    for file in folder.rglob("*.csv"):
        try:
            # Read only the first 5 rows to get column names and a preview
            df = pd.read_csv(file, nrows=5)
            meta = {
                "filename": file.name,
                "path": file,
                "columns": [c.lower() for c in df.columns],
                "shape": df.shape,
                "preview": df.head().to_string(index=False)
            }
            globals_space._DATASET_METADATA[file.name] = meta
            globals_space.logger.info(f"Dataset metadata generated of file: {file.name}")
        except Exception as e:
            globals_space.logger.warning(f"Warning reading {file}: {e}")
            continue

    globals_space.logger.info("Metadata is succesfully generated!.")


def select_relevant_datasets(user_input: str) -> dict:
    """
    Given a user's question, suggest the CSV file most likely to contain the answer. Scores are computed by keyword overlap between the question and column names. Returns a dict with filename and summary metadata.
    """
    globals_space.logger.info("\nselect_relevant_dataset function called!\n")

    # Ensure metadata is built
    if not globals_space._DATASET_METADATA:
        build_dataset_metadata()

    # If still no metadata after attempting to build, return empty dict
    if not globals_space._DATASET_METADATA:
        return {}

    
    best_match = None
    best_score = -1
    text = user_input.lower()
    
    # First check if the user question mentions the filename explicitly
    for filename, meta in globals_space._DATASET_METADATA.items():
        name_no_ext = filename.lower().replace(".csv", "")
        if name_no_ext in text:
            globals_space.logger.info("\n[EXPLICIT DATASET MATCH FOUND]")
            best_match = meta
            best_score = 9999   # High score for filename match
            break
    
    # Otherwise, do keyword matching
    if best_match is None:
        tokens = set(user_input.lower().split())
        # Deside what metadata to check. Here is checked both columns and filename.
        for meta in globals_space._DATASET_METADATA.values():
            score = sum(1 for t in tokens if t in meta["columns"] or t in meta["filename"].lower())
            if score > best_score:
                best_score = score
                best_match = meta

    if best_match:
        # Logging the best match details for debugging
        globals_space.logger.info("\nResult of select_relevant_dataset_tool:")
        globals_space.logger.info(f"Best matching dataset for the question: '{user_input}'")
        globals_space.logger.info(f"Filename: {best_match['filename']}")
        globals_space.logger.info(f"Path: {best_match['path']}")
        globals_space.logger.info(f"Columns: {best_match['columns']}")
        globals_space.logger.info(f"Shape: {best_match['shape']}")
        globals_space.logger.info(f"Preview: \n{best_match['preview']}\n")
        return best_match
    else:
        globals_space.logger.info("No relevant dataset found.")
        return "No dataset is available for analysis. The chatbot can still help you with general questions, but data-driven analysis is not possible at this time."    


def auto_analyse(user_input: str) -> str:
    """
    Automatically pick the best dataset and use PandasAI to answer a question. Returns a summary and the analysis result.
    """
    globals_space.logger.info("\nauto_analyse function called.\n")

    meta = select_relevant_datasets(user_input)
    if not meta:
        return "No relevant dataset found."
    
    # PandasAI reads the chosen CSV as a "Smart" Dataframe
    df = pai.read_csv(meta["path"])

    # Summary helps with debugging and transparency
    summary = (
        f"Chosen dataset: {meta['filename']}\n"
        f"Columns: {', '.join(df.columns)}\n"
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Preview:\n{meta['preview']}\n"
    )
    try:
        answer = df.chat(user_input)    # PandasAI generates analysis code
        globals_space.logger.info(f"\nResult of auto_analyse:\nSummary:\n{summary}\nAnswer:\n{answer}\n")
        return summary + "\nAnalysis Result:\n" + str(answer)
    except Exception as e:
        # If PandasAI fails, retrurn the error message instead of crashing the application
        return summary + "\nAnalysis error: " + str(e)


def retrieve_context(query: str):
    """
    Retrieve relevant context from the RAG vectorstore. Returns formatted text that can be used to answer the users query.
    """
    globals_space.logger.info("retrieve_context function called.\n")
    
    # If retriever is not initialized, we cannot search the vector store
    if globals_space._RETRIEVER is None:
        return "I cannot retrieve context because the retriever is not initialized. Please set up the vector store first."
    
    docs = globals_space._RETRIEVER.invoke(query)
    if not docs:
        return "No dataset is available for context retrieval. However, I can still assist you with general questions about waste management."

    context_texts = []
    globals_space.logger.info("Retrieved context:\n")
    for doc in docs:
        # doc.page_content is the text chunk, doc.metadata descrbes its origin
        globals_space.logger.info(f"→ {doc.page_content}")
        globals_space.logger.info(f"Metadata: {doc.metadata}")
        context_texts.append(
            f"{doc.page_content}\n(Metadata: {doc.metadata})"
        )

    return context_texts

# LangChain requires the @tool wrapper so the agent can call the function.
# The agent cannot directly call normal python functions unless they are tools.
@tool
def auto_analyse_tool(user_input: str) -> str:
    """
    Automatically pick the best dataset and use PandasAI to answer a question. Returns a summary and the analysis result.
    """
    result = auto_analyse(user_input)
    globals_space.logger.info(f"\nResult of auto_analyse: {result}\n")
    return result

@tool
def retrieve_context_tool(user_input: str):
    """
    Retrieve relevant context from the RAG vectorstore. Returns formatted text that can be used to answer the users query.
    """
    result = retrieve_context(user_input)
    globals_space.logger.info(f"\nResult of retrieve_context_tool: {result}\n")
    return result