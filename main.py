import os
import pathlib
import langchain
# from langchain_ollama import ChatOllama
# from langchain.tools import tool
from langchain.agents import create_agent
import langchain_ollama
import langchain_community
from langchain_chroma import Chroma

# import tools
from rag import load_csvs_from_folder, clean_id_column, normalize_id_series, clean_dataframes_ids, load_all_csvs, build_summary_chunks, build_vector_store, get_retriever
from tools import retrieve_context_tool, auto_analyse_question_tool, select_relevant_dataset_tool, retrieve_context, auto_analyse_question, select_relevant_dataset

import globals_space
from globals_space import _VECTOR_STORE, _RETRIEVER, _DATA_FOLDER, _DATASET_METADATA

SYS_PROMPT = """
You are a chatbot agent for the company Waste Vision. This company is specializing in the development of sustainable solutions for waste management.
Your task is to assist users by answering their questions based on the available waste management data.
The datasets describe containers, locations, devices, routes and operational events in the form of .CSV files. These datasets together describe the real-world waste collection process, linking physical assets (containers, devices, locations) with operational events (collections, maintatanance and sensor readings). 
Your role as the LLM agent is to use the available tools to analyze, explain, and answer questions about container usage, fill levels, and factual sources.
Available tools:
- 'select_relevant_dataset_tool':  
   Use this tool to scan all available CSV files and identify which dataset is most relevant to the user’s question.  
   This is especially useful when the user does not mention a specific file.
- 'auto_analyze_question_tool':  
   Use this tool when the user is asking for data exploration, counts, trends, or statistics based on raw CSV data.  
   This tool will automatically select the correct dataset and run Python code (via PandasAI) to answer the question.  
   Do not use this tool for high-level container descriptions — use the Retrieval Augmented Generation tool for that instead.
Always choose the tool that best fits the question. Use `auto_analyze_question` for calculations.
When answering questions, ensure that you:
1. Read the question carefully.
2. Choose the tool that provides you the best context to answer the question.
3. Use the provided context from the tool to answer the question.
4. If the information is not available to you or you are not sure. Don't halucinate, but explain that you don't know.
5. Maintain a proffessional and helpful tone.
"""
# - 'retrieve_context_tool': Retrieval Augmented Generation context provider that searches a vector database of the .CSV files, based on the user's input to retrieve contextual information.
#  Use `retrieve_context` for descriptive facts.

# Set definitions
EMBEDDING_MODEL = langchain_ollama.OllamaEmbeddings(model="mxbai-embed-large:335m")
VECTOR_STORE_PATH = "./chroma_location_embeddings"

AGENT_MODEL = langchain_ollama.ChatOllama(
    model="gpt-oss:20b",
    validate_model=True,
    temperature=0.3
)
# Amount of context chunks the agent will recieve from vector database
CONTEXT_AMOUNT = 1



LLM_AGENT = create_agent(
    model = AGENT_MODEL,
    system_prompt = SYS_PROMPT, #TODO: include sys prompt
    tools=[select_relevant_dataset_tool, auto_analyse_question_tool]   #TODO: include tools ->>     retrieve_context_tool
)


def rag():    # Pre-load data or perform any necessary setup here
    global _VECTOR_STORE
    global _RETRIEVER

    print("Pre-processing: Loading waste management data...")
    df = load_all_csvs(_DATA_FOLDER)
    
    print("Building summary chunks...")
    chunks = build_summary_chunks(df)

    print("Building vector store...")
    _VECTOR_STORE = build_vector_store(chunks, EMBEDDING_MODEL, VECTOR_STORE_PATH)

    _RETRIEVER = _VECTOR_STORE.as_retriever(search_kwargs={"k": 3})

    return _VECTOR_STORE

def pandas_chatbot_run():  #(vector_store):
    # print("Setting up retriever...")
    # retriever = get_retriever(vector_store, k=1)
    
    print("Type 'exit' to end the session.")
    while True:
        user_input = input("\nEnter your prompt: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Exiting.")
            break

        # docs = retriever.invoke(user_input) # Get relevant documents

        # for doc in docs:
        #     print("→", doc.page_content)
        #     print("Metadata:", doc.metadata)
        
        # rag_prompt = f"User Input: {user_input}\nRelevant Context: {docs}\n Answer the user's question based on the above context."
        
        tools_prompt = f"User Input: {user_input}\nUse relevant tools and user input to answer the users question"

        # agent.invoke expects the messages structure used earlier
        response = LLM_AGENT.invoke(
            {"messages": [{"role": "user", "content": tools_prompt}]})

        # Print the agent response (message / text)
        print("\nResponse:\n", response)



def rag_chatbot_run():
    print("Rag Chatbot running:...")

    # Check if vector store is present in file system
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Vector store found at {VECTOR_STORE_PATH}\n")
        globals_space._VECTOR_STORE = Chroma(
            persist_directory=VECTOR_STORE_PATH, 
            embedding_function=EMBEDDING_MODEL,
            collection_name="location_summaries")
    else:
        print(f"Vector store not found at {VECTOR_STORE_PATH}. Building new vector store...")
        VECTOR_STORE = rag()

    globals_space._RETRIEVER = get_retriever(globals_space._VECTOR_STORE, CONTEXT_AMOUNT)
        

    while(True):
        user_input = input("\nEnter your prompt: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Exiting.")
            break
        
        # docs = retriever.invoke(user_input)
        # print("Retrieved context:\n")
        # for doc in docs:
        #     print("→", doc.page_content)
        #     print("Metadata:", doc.metadata)
        
        # prompt = f"Here is the question from the user that you need to answer:{user_input}\nTo answer the question, use the following context:{docs}\n"

        response = LLM_AGENT.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )

        print(response)


def main():
    #rag()
    #pandas_chatbot_run()
    #rag_chatbot_run()
    #build_dataset_metadata()
    result = auto_analyse_question("How many container locations are listed in the designated location export?")
    print(result)

if __name__ == "__main__":
    main()

