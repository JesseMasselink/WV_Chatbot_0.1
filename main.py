# chatbot/main.py
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from langchain.tools import tool
from tools.csv_tools import load_waste_management_data

SYSTEM_PROMPT = """
You are an expert in answering questions about waste management.
You have access to a tool called 'load_waste_management_data' that allows you to load and summarize waste management data from CSV files. Use this tool whenever you need to analyze or extract information from CSV files related to waste management.
"""

model = ChatOllama(model="llama3.2", temperature=0.3)

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[load_waste_management_data],
)

def chatbot_run():
    print("Type 'exit' to end the session.")
    while True:
        user_input = input("\nEnter your prompt: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Exiting.")
            break

        # agent.invoke expects the messages structure used earlier
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            context={},
        )

        # Print the agent response (message / text)
        print("\nResponse:\n", response)

if __name__ == "__main__":
    chatbot_run()
