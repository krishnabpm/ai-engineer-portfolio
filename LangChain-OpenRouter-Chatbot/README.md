# LangChain OpenRouter Chatbot

A command-line chatbot built with LangChain and [OpenRouter](https://openrouter.ai), using free models.

## Features
- Streams the reply word by word as it is generated
- Remembers the whole conversation (chat history)
- Automatic fallback to backup free models if the main one is busy
- 60-second timeout and 1 retry, so it never hangs forever

## Setup
```bash
cd LangChain-OpenRouter-Chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENROUTER_API_KEY
```

## Run
```bash
python 1.ChatModels/openrouter_streaming_chatbot.py
```
Type `exit` to quit.
