import gradio as gr
from PIL import Image
import math
import io
import argparse
import torch
import os
import json
import torch
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


"""
from googletrans import Translator
from gtts import gTTS
import os

def translate_text(text, src_lang, dest_lang):
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest=dest_lang)
    return translation.text

def text_to_speech(text, lang, filepath):
    tts = gTTS(text=text, lang=lang)
    tts.save(filepath)

# Example usage
arabic_text = "مثال على نص عربي"  # Replace this with your text from Whisper
english_text = translate_text(arabic_text, 'ar', 'en')
translated_arabic_text = translate_text(english_text, 'en', 'ar')

audio_file_path = "output.mp3"
text_to_speech(translated_arabic_text, 'ar', audio_file_path)
print(f"Arabic speech saved to {audio_file_path}.")


import whisper

def transcribe_audio(file_path, model_type="tiny"):
    model = whisper.load_model(model_type)
    result = model.transcribe(file_path,language="en")
    return result["text"]

# Example usage
audio_file_path = "output.mp3"  # Replace with your audio file path
transcription = transcribe_audio(audio_file_path)
print(transcription)

"""

### My main model....
model_path = '/media/pc/e/2025/Video-LLaVA/checkpoints/llava-7b-llama3-PMC-Instruction-lora'
model_base = 'aaditya/OpenBioLLM-Llama3-8B'


#conv_mode = 'llava_llam2'
conv_mode = 'llama_3'

disable_torch_init()
model_path = os.path.abspath(model_path)
print('model path')
print(model_path)
model_name = get_model_name_from_path(model_path)
print('model name')
print(model_name)
print('model base')
print(model_base)

tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name,
                                                                 device='cuda:0')  # print(args.question_file)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Replace 'YOUR_API_KEY' with your actual OpenAI API key

# Initialize the conversation history as an empty list
conversation_history = []
chatbot_response=[]
image_mem = None
# Function to use GPT-3 as the chatbot
def clear_history(chat_history):
    global conversation_history, chatbot_response, image_mem
    conversation_history = []
    chatbot_response = []
    image_mem = None
    return [], None
def add_text(text,chat_history):
    chat_history.append((text, None))
    return '', chat_history
def add_file(chat_history, file):
    global image_mem
    chat_history.append(((file.name,), None))
    #process the images here
    image_mem = Image.open(file.name)
    return chat_history

def chat_with_gpt(text, chat_history):
    global conversation_history,chatbot_response, image_mem # Use the global conversation_history variable
    # Prepare the input text, adding image-related tokens if needed

    image_processor = processor['image']
    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    cur_prompt = chat_history[-1][0]
    #if not image_mem:
    #    chat_history.append((None, "Upload an image first so you can ask about it 😊"))
    #    return "", chat_history
    if image_mem:
        image_tensor = image_processor.preprocess(image_mem, return_tensors='pt')['pixel_values']
        if type(image_tensor) is list:
            tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            tensor = image_tensor.to(model.device, dtype=torch.float16)


    print(f"{roles[1]}: {cur_prompt}")
    cur_prompt = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt
    conv.append_message(conv.roles[0], cur_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(prompt)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if image_mem:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
    else:

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])


    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    print(outputs)
    # print(outputs)
    response = outputs.split('|<end_of_text>|')[0]

    chat_history.append((None, response))
    return "", chat_history





def example_add(msg, file,chat_history):
    global image_mem
    image_mem = Image.open(file)
    chat_history.append(((file,), None))
   # process the images here
   #  add_text(msg, chat_history)
   #  a, chat_history = chat_with_gpt(msg, chat_history)
    return chat_history



title = "KSU-Agent (1)"
description = ("Interact with the AI Agent by uploading an image or a video and asking questions. "
               "The AI will respond based on the image and your query. The model can use external tools to answer your questions")

# Adjusting the interface layout
with gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=False) as demo:
    with gr.Column():

        # Image upload column on the left, smaller size
        with gr.Row():
            with gr.Column():
                gr.Markdown("<img src='https://kscdr.org.sa/sites/default/files/2021-08/KSU-1.png'>",scale=3)
                gr.Markdown(f"# {title}\n{description}", scale=3)
                curr_img = gr.Image(label="Uploaded Image", interactive=False, type="filepath", scale=6)
                btn = gr.UploadButton("Click to upload Image 🖼️", file_types=["image"] )
                clear = gr.Button("Clear Chat 🗑️")

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Dialogue", height=800)
                msg = gr.Textbox(label="Enter your question here", placeholder="Type your question and hit enter")
                sub = gr.Button("Submit")
        # Chat window column on the right, larger size
        ex = gr.Examples([['What is this?', 'ex_images/1.jpg'],
                              ['Describe image?', 'ex_images/2.webp'],
                              ['What is this?', 'ex_images/3.webp']]
                             , [msg, curr_img, chatbot], chatbot, example_add, run_on_click=True)

    # Bind functions to widgets
    msg.submit(add_text, [msg, chatbot], [msg, chatbot]).then(fn=chat_with_gpt,inputs=[msg,chatbot], outputs=[msg,chatbot])
    clear.click(clear_history, [], [chatbot, curr_img])
    btn.upload(add_file, [chatbot, btn], [chatbot]).then(lambda a: image_mem.resize((int(image_mem.size[0]*300/image_mem.size[1]), 300)), outputs=[curr_img])
    sub.click(add_text, [msg, chatbot], [msg, chatbot]).then(fn=chat_with_gpt,inputs=[msg,chatbot], outputs=[msg,chatbot])
    # Optional: Add a footer for credits or additional information
    gr.Markdown("---\n© 2024 BigData@KSU Team: Computer Engineering Department-CCIS")

if __name__ == "__main__":
    demo.launch()
