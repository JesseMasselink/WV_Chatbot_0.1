import pandas as pd
import pandasai as pai
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM

from langchain.tools import tool

from rag import get_retriever
import globals_space
from globals_space import _RETRIEVER, _DATA_FOLDER, _DATASET_METADATA

import os
import pathlib
from pathlib import Path



# from langchain.tools import tool

pandasai_llm = LiteLLM(
    model = "ollama/gpt-oss:20b",
    api_key = "82fe034cdb664ebb936d7121397f664d.lf_w5tGConKuske3_nfHXLDo"
)

pai.config.set({
    "llm": pandasai_llm,
    
})


def build_dataset_metadata():
    """Populate the DATASET_METADATA dict by scanning all available CSV files"""

    print("\nTEST: Build_dataset_metadata function called!\n")

    global _DATASET_METADATA

    folder = Path(_DATA_FOLDER)
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")

    for file in folder.rglob("*.csv"):
        try:
            df = pd.read_csv(file, nrows=5)
            meta = {
                "filename": file.name,
                "path": file,
                "columns": [c.lower() for c in df.columns],
                "shape": df.shape,
                "preview": df.head().to_string(index=False)
            }
            _DATASET_METADATA[file.name] = meta
            print("Dataset metadata generated.")
        except Exception as e:
            print(f"ERROR reading {file}: {e}")
            continue


def select_relevant_dataset(user_input: str) -> dict:
    """
    Given a user's question, suggest the CSV file most likely to contain the answer.
    Scores are computed by keyword overlap between the question and column names.
    Returns a dict with filename and summary metadata.
    """
    print("\nTEST: select_relevant_dataset function called!\n")
    global _DATASET_METADATA
    if not _DATASET_METADATA:
        build_dataset_metadata()

    tokens = set(user_input.lower().split())
    best_match = None
    best_score = -1
    # Deside what metadata to send
    for meta in _DATASET_METADATA.values():
        score = sum(1 for t in tokens if t in meta["columns"] or t in meta["filename"].lower())
        if score > best_score:
            best_score = score
            best_match = meta
    return best_match or {}


def auto_analyse_question(user_input: str) -> str:
    """
    Automatically pick the best dataset and use PandasAI to answer a question.
    Returns a summary and the analysis result.
    """
    print("\nTEST: auto_analyse_question function called!\n")
    meta = select_relevant_dataset(user_input)
    if not meta:
        return "No relevant dataset found."
    df = pd.read_csv(meta["path"])
    summary = (
        f"Chosen dataset: {meta['filename']}\n"
        f"Columns: {', '.join(df.columns)}\n"
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Preview:\n{meta['preview']}\n"
    )
    # use existing pandasai_llm (already defined in tools.py)
    smart_df = SmartDataframe(df, config={"llm": pandasai_llm})
    try:
        answer = smart_df.chat(user_input)
        return summary + "\nAnalysis Result:\n" + str(answer)
    except Exception as e:
        return summary + "\nAnalysis error: " + str(e)




# def pandasai_test(query) -> str:
#     """
#     Test tool for PandasAI functionality.
#     Args:
#         query (str): The question to test.
        
        
#     Returns:
#         str: The result of the query.
#     """
    

#     file_df = pd.read_csv("/home/aiadmin/WasteVision/GAD2/device_fill_level/device_fill_level_export_edit.csv")

    

#     answer = df.chat(query)
#     return answer

def retrieve_context(query: str):
    """
    Retrieve relevant context from the RAG vectorstore.
    Returns formatted text that can be used to answer the users query.
    """
    print("RAG_TOOL in use!")
    
    if globals_space._RETRIEVER is None:
        return "ERROR: Retriever not initialized."
    
    docs = globals_space._RETRIEVER.invoke(query)

    context_texts = []

    print("Retrieved context:\n")
    for doc in docs:
        print("→", doc.page_content)
        print("Metadata:", doc.metadata)
        context_texts.append(
            f"{doc.page_content}\n(Metadata: {doc.metadata})"
        )

    return context_texts


#Langchain tool Wrappers > This is needed so that the functions can be called seperately. Langchain does not allow that if it is a @tool.

@tool
def select_relevant_dataset_tool(user_input: str) -> dict:
    """
    Given a user's question, suggest the CSV file most likely to contain the answer.
    Scores are computed by keyword overlap between the question and column names.
    Returns a dict with filename and summary metadata.
    """
    dict_result = select_relevant_dataset(user_input)
    for d in dict_result:
        print("\nResult of select_relevant_dataset_tool: ", d, "\n")
    return dict_result

@tool
def auto_analyse_question_tool(user_input: str) -> str:
    """
    Automatically pick the best dataset and use PandasAI to answer a question.
    Returns a summary and the analysis result.
    """
    result = auto_analyse_question(user_input)
    print("\nResult of auto_analyse_question: ", result, "\n")
    return result

@tool
def retrieve_context_tool(user_input: str):
    """
    Retrieve relevant context from the RAG vectorstore.
    Returns formatted text that can be used to answer the users query.
    """
    result = retrieve_context(user_input)
    print("\nResult of auto_analyse_question: ", result, "\n")
    return result