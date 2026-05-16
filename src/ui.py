import gradio as gr
import time
from src.assistant import chat

def stream_msg(message, history):
    for i in range(len(message)):
        time.sleep(0.3)
        yield message[: i+1]

demo = gr.ChatInterface(
    fn=lambda msg, hist: chat(msg),
    title="Soccer Tactical Assistant",
    description="Ask anything about formations, tactics, and strategy.",
    examples=[
        "Explain gegenpressing and who pioneered it",
        "What are the trade-offs of a high defensive line?",
        "How does Guardiola's positional play work?"
    ],
    fill_height=True,
    autofocus=True
)

if __name__ == "__main__":
    demo.launch(
        theme="ocean",
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )   # opens automatically at http://localhost:7860