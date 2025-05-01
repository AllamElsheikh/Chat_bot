import gradio as gr

# Initial welcome message
chat_history = [("🤖 Bot", "👋 Hello there! I'm FCI-GPT your assistant. Ask me anything.")]

def chatgpt_ui(message):
    # Format prompt
    prompt = "Human: " + message + "\nAssistant:"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_length=256)
    response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

    # Extract only new bot reply
    bot_reply = response[len(prompt):].strip()

    # Add user and bot messages
    chat_history.append(("🧑 You", message))
    chat_history.append(("🤖 Bot", bot_reply))
    return chat_history

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'> FCI-GPT</h1>")

    chatbot = gr.Chatbot(value=chat_history)
    msg = gr.Textbox(placeholder="Type your message here...", label="")

    def user_input(message):
        return chatgpt_ui(message)

   # msg.submit(user_input, msg, chatbot)
    msg.submit(user_input, msg, chatbot).then(lambda: "", None, msg)

demo.launch(share=True)
