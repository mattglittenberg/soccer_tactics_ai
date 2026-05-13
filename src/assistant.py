# assistant.py
import ollama
from prompts import SYSTEM_PROMPT
from rag import retrieve, build_augmented_prompt
from config import MODEL, N_RETRIEVAL_RESULTS

history = []

def chat(user_message: str, use_rag: bool = True) -> str:
    if use_rag:
        chunks = retrieve(user_message, n_results=N_RETRIEVAL_RESULTS)
        message_to_send = build_augmented_prompt(user_message, chunks)

        # Show what was retrieved (helpful during development)
        print(f"\n📖 Retrieved {len(chunks)} chunks:")
        for c in chunks:
            print(f"   {c['source']} (relevance: {c['relevance']})")
    else:
        message_to_send = user_message

    history.append({"role": "user", "content": message_to_send})

    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        stream=True
    )

    full_reply = ""
    print("\nCoach: ", end="", flush=True)
    for chunk in stream:
        text = chunk["message"]["content"]
        print(text, end="", flush=True)
        full_reply += text

    print("\n")

    # Store the original user message in history (not the augmented one)
    # This keeps the conversation context clean
    history[-1] = {"role": "user", "content": user_message}
    history.append({"role": "assistant", "content": full_reply})

    return full_reply


if __name__ == "__main__":
    print("⚽ Soccer Tactical Assistant (RAG enabled) — type 'quit' to exit\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        chat(user_input)