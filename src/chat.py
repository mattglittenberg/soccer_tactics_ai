import ollama
from prompts import SYSTEM_PROMPT
from config import MODEL

history = []

def chat(user_message):
    history.append({"role": "user", "content": user_message})
    
    chat_stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        stream=True
    )

    full_reply = ""
    print("\nCoach: ", end="", flush=True)
    for chunk in chat_stream:
        chat_txt = chunk["message"]["content"]
        print(chat_txt, end="", flush=True)
        full_reply += chat_txt

    print("\n")
    history.append({"role": "assistant", "content": full_reply})


if __name__ == "__main__":
    print("⚽ Soccer Tactical Assistant — type 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        chat(user_input)
