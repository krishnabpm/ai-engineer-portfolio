from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

import os

load_dotenv()

model = ChatOpenAI(
    model="google/gemma-4-31b-it:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    # Give up after 60 seconds instead of waiting forever
    timeout=60,
    max_retries=1,
    # Backup free models, tried in order if the main one is busy
    extra_body={"models": [
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ]}
)

# Keeps the whole conversation so the model remembers earlier messages
chat_history = []

print("Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    # Show a loading message until the first words arrive
    print("AI: thinking...", end="", flush=True)

    reply = ""
    try:
        # stream() sends the reply piece by piece, so you see it being written
        for chunk in model.stream(chat_history):
            # Skip empty pieces (the model sends these while it is still thinking)
            if not chunk.content:
                continue
            # Replace "thinking..." with the reply once the first words arrive
            if not reply:
                print("\rAI:            \rAI: ", end="", flush=True)
            reply += chunk.content
            print(chunk.content, end="", flush=True)
    except Exception as e:
        print(f"\rAI: Error - {e}")
        chat_history.pop()
        continue

    print("\n")

    chat_history.append(AIMessage(content=reply))
