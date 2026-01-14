# main.py
"""
MAIN APPLICATION (Streamlit UI + LLM Agent Runner)

This file is the entry point of the application.
It does three main things:

1) Defines the "system prompt" (rules + role) for the chatbot.
2) Runs the LLM agent (the model + tool calling logic).
3) Creates the Streamlit UI so users can chat in a browser.

High-level flow:
User types a question -> LLM_agent_run() -> agent may call a tool -> agent returns final answer -> Streamlit displays it.
"""

# Standard libraries
import os

# Langchain and Ollama for LLM and agent creation
from langchain.agents import create_agent
from langchain_chroma import Chroma

# Streamlit for UI
import streamlit as st

# Import tools
from rag import load_all_csvs, build_summary_chunks, build_chroma_vector_store, get_retriever
from tools import retrieve_context_tool, auto_analyse_tool, build_dataset_metadata
import globals_space


# System prompt for the Waste Vision Chatbot agent
# It defines the instruction text that the LLM agent will follow when answering user queries.
SYS_PROMPT = """
You are a chatbot agent for the company Waste Vision. Waste Vision specializes in developing sustainable solutions for waste management.
Your task is to assist users (clients of Waste Vision) by answering their questions based on the available waste management data of that specific client.
The client datasets describe containers, locations, devices, routes, and operational events in the form of .CSV files. Together, these datasets describe the real-world waste collection process by linking physical assets (containers, devices, locations) with operational events (collections, maintenance, and sensor readings).

If the query requires external information to answer the question, you must use the available tools to analyze, explain, and answer questions about container usage, fill levels, and factual sources.
- retrieve_context_tool: Retrieval Augmented Generation (RAG) context provider that searches a vector database of .CSV files based on the user's input. Use retrieve_context for descriptive facts about containers (high-level descriptions).
- auto_analyse_tool: Use this tool when the user asks for data exploration, counts, trends, or statistics based on raw CSV data.
This tool will automatically select the correct dataset and run Python code (via PandasAI) to answer the question. Do not use this tool for high-level container descriptions, use the RAG tool for that instead.

Tool selection rule:
Always choose only one tool per user question, and only if it is needed. Pick the tool that best matches the question:
- Descriptive / explanatory container facts → retrieve_context_tool
- Numeric analysis / trends / statistics → auto_analyse_tool

You also have access to the full conversation history. When answering questions:
- If the user asks a similar or related question again, refer to your previous answer or analysis if it's relevant
- Use context from earlier messages to provide consistent and coherent responses
- If the current question relates to a previous topic, acknowledge that connection and build upon it

Answering procedure:
When answering questions, you must:
- Read the question carefully and identify whether it needs descriptive context or data analysis.
- Choose one tool that provides the best context for the question (or no tool if not needed).
- Use the tool output as the basis of your answer.
- If the information is not available or you are not sure, do not hallucinate. Clearly say you do not know based on the available information.
- Maintain a professional and helpful tone.
"""

# Create the LLM agent with the specified model, system prompt, and tools
LLM_AGENT = create_agent(
    model = globals_space._AGENT_MODEL,
    system_prompt = SYS_PROMPT,
    tools=[retrieve_context_tool, auto_analyse_tool]
)

# Function to build the vector store
def build_vector_store():
    """Build the vector store from the waste management CSV dataset."""
    globals_space.logger.info("Build_vector_store function called.\n")

    # Load and merge all CSV data into one Dataframe
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

# Function to run the LLM agent
def LLM_agent_run(conversation: str, user_input: str) -> str:
    """
    Runs the agent on a singe user question and returns the response.
    """
    globals_space.logger.info("\nChatbot running:...\n")

    # If the vector store folder exists, load it (fast startup).
    # Otherwise, try to build it (slower because embeddings must be generated).
    # If both fail, continue without RAG capabilities.
    if os.path.exists(globals_space._VECTOR_STORE_PATH):
        try:
            print(f"Vector store found at {globals_space._VECTOR_STORE_PATH}\n")
            globals_space._VECTOR_STORE = Chroma(
                persist_directory=globals_space._VECTOR_STORE_PATH, 
                embedding_function=globals_space._EMBEDDING_MODEL,
                collection_name="location_summaries")
            globals_space._RETRIEVER = get_retriever(globals_space._VECTOR_STORE, globals_space._CONTEXT_AMOUNT)
        except Exception as e:
            globals_space.logger.warning(f"Could not load vector store: {e}. Continuing without RAG.")
            globals_space._VECTOR_STORE = None
            globals_space._RETRIEVER = None
    else:
        try:
            globals_space.logger.info(f"Vector store not found at {globals_space._VECTOR_STORE_PATH}. Building new vector store...")
            globals_space._VECTOR_STORE = build_vector_store()
            globals_space._RETRIEVER = get_retriever(globals_space._VECTOR_STORE, globals_space._CONTEXT_AMOUNT)
        except Exception as e:
            globals_space.logger.warning(f"Could not build vector store: {e}. Continuing without RAG.")
            globals_space._VECTOR_STORE = None
            globals_space._RETRIEVER = None

    # Run the agent. The agent may call a tool depending on the question.
    try:
        # Build conversation context from interaction history
        conversation_context = ""
        if conversation:
            conversation_context = "Previous conversation (use this as context if needed):\n"
            for msg in conversation:
                role = msg["role"].upper()
                content = msg["content"]
                conversation_context += f"{role}: {content}\n"
            conversation_context += "\n---\n\n"
        
        # Combine context with current question
        full_input = conversation_context + f"Current question to answer: {user_input}"
        
        # Invoke agent with the combined input
        agent_result = LLM_AGENT.invoke({"messages": {"role": "user", "content": full_input}})

    except Exception as e:
        # Running the agent can fail if Ollama is not reachable or the model is not installed.
        return (
            "Sorry, I could not reach the local language model selected in the configuration. "
            "Please ensure that Ollama is running and the required model is installed. "
            "Error details: " + str(e)
            )

    # LangChain returns a message list, this gets the last message as the response
    ai_message = agent_result["messages"][-1]
    response = ai_message.content

    return response

# Streamlit UI configuration
def configure_page() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Waste Vision Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

# Streamlit chatbot application loop
def streamlit_chatbot():
    """Run the Streamlit chatbot application."""
    configure_page()
    st.title("Waste Vision Chatbot 🤖")

    # Build dataset metadata used by the auto_analyse tool
    if "metadata_ready" not in st.session_state:
        try:
            build_dataset_metadata()
            st.session_state["metadata_ready"] = True
        except Exception as e:
            globals_space.logger.warning("Warning: Could not build dataset metadata: {e}")
            st.warning(f" Dataset not available: {e}")
            st.session_state["metadata_ready"] = True
            return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response from LLM agent
        response = LLM_agent_run(st.session_state.messages, prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chatbot response in chat
        with st.chat_message("assistant"):
            st.markdown(response.title())

# Main entry point
def main():
    streamlit_chatbot()

if __name__ == "__main__":
    main()

