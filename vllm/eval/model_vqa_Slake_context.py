import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device='cuda:0')



    with open(args.question_file, 'r', encoding='utf-8') as file:
        dataset_1 = json.load(file)
    dataset_1=dataset_1['slake']

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    results_all=[]
    OA_closed=[]
    for iter1 in range(len(dataset_1)):

        #### pathVQA
        #image_file = '/media/pc/e/2025/BigData/Med_MEGA/finetune_images/PathVQA/test/' + dataset_1[iter1]['image'] + '.jpg'

        ### Salke
        image_file = '/media/pc/d/2025/BigData/Med_MEGA/finetune_images/slake/imgs/' + dataset_1[iter1]['image']


        results={ 'image':dataset_1[iter1]['image'],
                  'pred_answer':[],
                  'answer': [],
                  'summary':[],
                  'summary_pred':[]

                  }

        print('*********************************************************')
        print('Image Analysis',image_file)

        previous_context=' '

        for iter2 in range(len(dataset_1[iter1]['questions'])+1):

            if iter2<len(dataset_1[iter1]['questions']):
                cur_prompt_without_image = dataset_1[iter1]['questions'][iter2]
                if iter2==0:
                    previous_context=cur_prompt_without_image
                else:
                    cur_prompt_without_image=previous_context+cur_prompt_without_image

            else:
                cur_prompt_without_image='Describe the image.'
                cur_prompt_without_image = previous_context + cur_prompt_without_image

            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            cur_prompt=cur_prompt_without_image
            print(cur_prompt)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            image = Image.open(image_file).convert('RGB')
            image_processor = processor['image']
            conv_mode = "llama_3"
            conv = conv_templates[conv_mode].copy()
            #roles = conv.roles
            #print(image_file)

            #print(f"{roles[1]}: {cur_prompt}")

            cur_prompt = DEFAULT_IMAGE_TOKEN + '\n' + cur_prompt
            conv.append_message(conv.roles[0], cur_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            #print(prompt)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).to('cuda:0')

            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            #print(image_tensor.shape)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.to('cuda:0'),
                        do_sample=True,
                        temperature=0.1,
                        max_new_tokens=2048,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            response = outputs.split('<|end_of_text|>')[0]

            print(response)

            if iter2<len(dataset_1[iter1]['questions']):
                results['pred_answer'].append(response)
                results['answer'].append(dataset_1[iter1]['answer'][iter2])
            else:
                results['summary'].append(dataset_1[iter1]['summary'])
                results['summary_pred'].append(response)

            previous_context=cur_prompt_without_image+response

            #if iter2 % 1 ==0:
            #    previous_context=' '


        print('Results over the image')
        print(results)
        print('######################################')
        results_all.append(results)
        print('*********************************************************')

    with open(answers_file, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=4)  # Use indent=4 for pretty printing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/media/pc/e/2025/Video-LLaVA/checkpoints/Slake/llava-7b-llama3-Slake-Dataset-NEW-Instruction-lora")
    parser.add_argument("--model-base", type=str, default='aaditya/OpenBioLLM-Llama3-8B')
    parser.add_argument("--image-folder", type=str, default="")

    #### Slake
    parser.add_argument("--question-file", type=str, default="/media/pc/d/2025/BigData/Med_MEGA/finetune_images/slake/Test_Slake_summary_Final.json")
    parser.add_argument("--answers-file", type=str, default="/media/pc/d/2025/BigData/Med_MEGA/finetune_images/slake/Slake_Predictions_Model_videollava_ROCO_BLIP-04June_Context.json")

    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    eval_model(args)


