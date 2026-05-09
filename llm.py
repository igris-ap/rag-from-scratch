"""
llm.py — Stage 1: Talk to Ollama directly via HTTP.

Ollama runs a local server at http://localhost:11434.
We just POST a JSON payload and read the response.
No SDK, no LangChain — plain Python + the built-in `urllib` library.
"""

import json
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration — change MODEL_NAME to match what you pulled
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------
# A message is a dict: {"role": "system"|"user"|"assistant", "content": "..."}

def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """
    Send a list of messages to Ollama and return the assistant's reply as a string.

    Args:
        messages:    List of {"role": ..., "content": ...} dicts.
                     Build this list yourself — it's just Python dicts.
        temperature: 0.0 = deterministic (good for RAG, always use this).
                     Higher = more creative/random.

    Returns:
        The assistant's reply text as a plain string.

    Raises:
        RuntimeError if Ollama isn't running or returns an error.
    """

    # Build the request payload — this is exactly what Ollama expects
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,          # False = wait for full response, not streaming
        "options": {
            "temperature": temperature,
        },
    }

    # Encode to JSON bytes (urllib needs bytes, not a string)
    body = json.dumps(payload).encode("utf-8")

    # Build and send the HTTP POST request
    request = urllib.request.Request(
        url=OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}.\n"
            f"Make sure Ollama is running: `ollama serve`\n"
            f"Original error: {e}"
        )

    # Parse the JSON response
    data = json.loads(raw)

    # Ollama's response looks like:
    # {
    #   "message": {"role": "assistant", "content": "...the reply..."},
    #   "done": true,
    #   ...
    # }
    return data["message"]["content"]


def chat_with_system(system_prompt: str, user_message: str) -> str:
    """
    Convenience wrapper: send a system prompt + one user message.
    This is the most common pattern in RAG — one system instruction,
    one user question.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    return chat(messages)


# ---------------------------------------------------------------------------
# Quick test — run this file directly to verify Ollama is working
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing Ollama connection...\n")

    # Test 1: simple question
    reply = chat_with_system(
        system_prompt="You are a helpful assistant. Keep answers to one sentence.",
        user_message="What is Retrieval-Augmented Generation?",
    )
    print(f"Reply: {reply}\n")

    # Test 2: multi-turn conversation (list of messages)
    conversation = [
        {"role": "user",      "content": "My name is Arjun."},
        {"role": "assistant", "content": "Hello Arjun! How can I help you?"},
        {"role": "user",      "content": "What is my name?"},
    ]
    reply2 = chat(conversation)
    print(f"Memory test: {reply2}\n")

    print("Stage 1 complete — Ollama is working.")
