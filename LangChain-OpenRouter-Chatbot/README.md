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
```

### Add your own API key
This repo does **not** include an API key. You must provide your own:

1. Get a free key at https://openrouter.ai/keys.
2. Copy the template and rename it to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and replace the placeholder with your own key:
   ```
   OPENROUTER_API_KEY=your-openrouter-api-key-here
   ```

`.env` is git-ignored, so your key is never committed. Never put a real key in `.env.example`.

## Run
```bash
python 1.ChatModels/openrouter_streaming_chatbot.py
```
Type `exit` to quit.

## Tests
The tests use a fake model, so they need no network and no API key.

```bash
pip install -r requirements-dev.txt
pytest
```

They cover the model settings (base URL, timeout, retries, fallback models), streaming (joining chunks, skipping empty ones), conversation memory, the `exit` command, and error handling (the failed message is dropped and the chat continues).
