import os
import gradio as gr
import openai

# Set your OpenAI API key via HF Secrets
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_SYSTEM = """
You are a helpful assistant for LearnDanishLab.com — a structured Danish learning platform for English speakers.
Answer questions about Danish grammar, vocabulary, pronunciation, and give learning tips.
If the user asks to chat in Danish, respond appropriately.
"""

def run_openai_chat(messages):
    """
    Send messages to OpenAI and return the model's response.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=600
    )
    text = response.choices[0].message["content"]
    return text

def api_chat(messages_json):
    """
    This function will be exposed as a clean API endpoint.
    It expects JSON like:
    { "messages": [{"role": "user", "content": "Hello"}] }
    """

    # Build full context
    msgs = [{"role": "system", "content": PROMPT_SYSTEM}]
    for m in messages_json.get("messages", []):
        msgs.append(m)

    reply = run_openai_chat(msgs)
    return {"reply": reply}

# -------------------------------
# Gradio Chat UI (optional)
# -------------------------------

def gradio_chat(history):
    """
    history: list of {"role": "...", "content": "..."}
    Return updated history after generating a reply.
    """
    # Build OpenAI messages from history
    msgs = [{"role": "system", "content": PROMPT_SYSTEM}]
    for msg in history:
        msgs.append(msg)

    assistant_reply = run_openai_chat(msgs)
    history.append({"role": "assistant", "content": assistant_reply})
    return history, None, None

def put_message_in_chatbot(message, history):
    """
    Adds user message to chat history
    """
    return "", history + [{"role": "user", "content": message}]

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
    with gr.Row():
        message = gr.Textbox(label="Ask about Danish or practice Danish:")

    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    ).then(
        gradio_chat,
        inputs=chatbot,
        outputs=[chatbot, None, None],
    )

ui.launch()
