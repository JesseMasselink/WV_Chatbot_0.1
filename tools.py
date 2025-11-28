import pandas as pd
import pandasai as pai
from pandasai import SmartDataframe
from pandasai_litellm.litellm import LiteLLM

from langchain.tools import tool

from rag import get_retriever
import globals_space
from globals_space import RETRIEVER

# from langchain.tools import tool

pandasai_llm = LiteLLM(
    model = "ollama/gpt-oss:20b",
    api_key = "82fe034cdb664ebb936d7121397f664d.lf_w5tGConKuske3_nfHXLDo"
)

pai.config.set({
    "llm": pandasai_llm,
    
})

def pandasai_test(query) -> str:
    """
    Test tool for PandasAI functionality.
    Args:
        query (str): The question to test.
        
        
    Returns:
        str: The result of the query.
    """
    

    file_df = pd.read_csv("/home/aiadmin/WasteVision/GAD2/device_fill_level/device_fill_level_export_edit.csv")

    df = SmartDataframe(
        file_df,
        config{
            "llm": pandasai_llm,
        }
    )

    answer = df.chat(query)
    return answer


@tool
def retrieve_context(query: str):
    """
    Retrieve relevant context from the RAG vectorstore.
    Returns formatted text that can be used to answer the users query.
    """
    print("RAG_TOOL in use!")
    
    if globals_space.RETRIEVER is None:
        return "ERROR: Retriever not initialized."
    
    docs = globals_space.RETRIEVER.invoke(query)

    context_texts = []

    print("Retrieved context:\n")
    for doc in docs:
        print("→", doc.page_content)
        print("Metadata:", doc.metadata)
        context_texts.append(
            f"{doc.page_content}\n(Metadata: {doc.metadata})"
        )

    return context_texts
