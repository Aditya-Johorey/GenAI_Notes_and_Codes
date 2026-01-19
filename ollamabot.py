import requests, json
from numpy.core.defchararray import lower

OLLAMA_API = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"

print("ChatBot Begins!😎")
print("The API can be slow. If no error thrown, pls be patient.")
print("type 'exit' to end chat")

conversation = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', '0', 'q']:
        break

    payload = {
        "model": MODEL,
        "messages": conversation,
        'stream': True
    }

    with requests.post(OLLAMA_API, json=payload, stream = True)  as r:
        print("Bot: ", end="", flush= True)
        full_reply = ""
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode())
                if "message" in data:
                    token = data["message"]["content"]
                    print(token, end="", flush=True)
                    full_reply += token
    print("\n")

    conversation.append({
        "role": "bot",
        "content": full_reply
    })
