import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/blenderbot-400M-distill"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

PERSONALITY = """
You are Doraemon, the friendly robotic cat from the 22nd century.
You help people with futuristic gadgets.
You are kind, cheerful and supportive.
You often mention Nobita.
Always introduce yourself as Doraemon if asked.
Keep responses friendly and simple.
"""

def chat(message, history):

    conversation = PERSONALITY + "\n"

    for user, bot in history:
        conversation += "User: " + user + "\n"
        conversation += "Doraemon: " + bot + "\n"

    conversation += "User: " + message + "\nDoraemon:"

    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Clean response
    if "Doraemon:" in response:
        response = response.split("Doraemon:")[-1].strip()

    return response


demo = gr.ChatInterface(
    fn=chat,
    title="🤖 Doraemon AI Chatbot",
    description="Hi! I am Doraemon from the 22nd century. Let's talk!",
    theme="soft"
)

demo.launch()
