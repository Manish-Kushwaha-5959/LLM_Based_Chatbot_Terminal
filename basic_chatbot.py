from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

chat_history = [
    SystemMessage(content="You are a helpful chatbot.")
]

def ask_bot(user_input):
    chat_history.append(HumanMessage(content=user_input))

    response = llm.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    return response.content


if __name__ == "__main__":
    print("Chatbot started! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        print("Bot:", ask_bot(user_input))
