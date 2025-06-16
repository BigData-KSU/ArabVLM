from PIL import Image
import os
import torch
from vllm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from vllm.conversation import conv_templates, SeparatorStyle
from vllm.model.builder import load_pretrained_model
from vllm.utils import disable_torch_init
from vllm.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


### Main model....
model_path ='/BigData-KSU/ArabVLM'
model_base = 'ALLaM-AI/ALLaM-7B-Instruct-preview'


conv_mode = 'llava_llama_2'
disable_torch_init()
model_path = os.path.abspath(model_path)
print('model path')
print(model_path)
model_name = get_model_name_from_path(model_path)
print('model name')
print(model_name)
print('model base')
print(model_base)

tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name,device='cuda:0')


def chat_with_Vision_BioLLM(cur_prompt,image_name):
    # Prepare the input text, adding image-related tokens if needed
    image_mem = Image.open(image_name).convert('RGB')
    image_processor = processor['image']
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    print(image_mem)
    image_tensor = image_processor.preprocess(image_mem, return_tensors='pt')['pixel_values']
    tensor = image_tensor.to(model.device, dtype=torch.float16)
    print(f"{roles[1]}: {cur_prompt}")
    cur_prompt = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt
    conv.append_message(conv.roles[0], cur_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if image_mem:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=tensor,
                do_sample=False,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])


    response = tokenizer.decode(output_ids[0, input_ids.shape[1]:])
    #print(outputs)

    return response


if __name__ == "__main__":

    cur_prompt='وصف الصورة بالتفصيل '
    image_name='/media/pc/e/2025/ArabVLM/sample_images/business/Tea.jpeg'
    outputs=chat_with_Vision_BioLLM(cur_prompt,image_name)
    print('Model Response.....')
    print(outputs)

