import argparse
import torch
import os
import json
from tqdm import tqdm


from PIL import Image
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)


# 图中是一名女子在沙滩上和狗玩耍，旁边是一只拉布拉多犬，它们处于沙滩上。

# 2nd dialogue turn
#response, history = model.chat(tokenizer, '框出图中击掌的位置', history=history)
#print(response)
# <ref>击掌</ref><box>(536,509),(588,602)</box>
#image = tokenizer.draw_bbox_on_latest_picture(response, history)
#if image:
#  image.save('1.jpg')
#else:
#  print("no box")



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    #model_path = os.path.expanduser(args.model_path)
    #model_name = get_model_name_from_path(model_path)
    #tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,device='cuda:0')
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use cuda device
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    with open(args.question_file, 'r', encoding='utf-8') as file:
        dataset_1 = json.load(file)
    dataset_1=dataset_1['PathVQA']

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    results_all=[]
    OA_closed=[]
    for iter1 in range(len(dataset_1)):

        #### pathVQA
        image_file = '/media/pc/d/2025/BigData/Med_MEGA/finetune_images/PathVQA/test/' + dataset_1[iter1]['image']+'.jpg'

        ### Salke
        #image_file = '/media/pc/d/2025/BigData/Med_MEGA/finetune_images/medvqa/images/' + dataset_1[iter1]['image']


        results={ 'image':dataset_1[iter1]['image'],
                  'pred_answer':[],
                  'answer': [],
                  'summary':[],
                  'summary_pred':[]

                  }

        print('*********************************************************')
        print('Image Analysis',image_file)

        for iter2 in range(len(dataset_1[iter1]['questions'])+1):

            cur_prompt='Give short answers the questions.For questions requring answers yes or no, just provide the answer yes or no. '

            if iter2<len(dataset_1[iter1]['questions']):
                cur_prompt = cur_prompt+dataset_1[iter1]['questions'][iter2]
            else:
                cur_prompt=cur_prompt+'Describe the image.'
            print(cur_prompt)
            image = Image.open(image_file).convert('RGB')

            # 1st dialogue turn
            query = tokenizer.from_list_format([
                {'image': image_file},
                # Either a local path or an url
                {'text': cur_prompt},
            ])
            response, history = model.chat(tokenizer, query=query, history=None)
            print(response)


            if iter2<len(dataset_1[iter1]['questions']):
                results['pred_answer'].append(response)
                results['answer'].append(dataset_1[iter1]['answer'][iter2])
            else:
                results['summary'].append(dataset_1[iter1]['summary'])
                results['summary_pred'].append(response)



        print('Results over the image')
        print(results)
        print('######################################')
        results_all.append(results)
        print('*********************************************************')

    with open(answers_file, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, indent=4)  # Use indent=4 for pretty printing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--model-path", type=str, default="/media/pc/e/2025/Video-LLaVA/checkpoints/medvqa/llava-7b-llama3-medvqa-Dataset-lora-BLIP-ROCO-5Epochs")
    #parser.add_argument("--model-path", type=str,default="/media/pc/e/2025/Video-LLaVA/checkpoints/all_dataset/llava-7b-llama3-Four-Dataset-lora-5Epochs")

    #parser.add_argument("--model-base", type=str, default='aaditya/OpenBioLLM-Llama3-8B')
    #parser.add_argument("--image-folder", type=str, default="")

    #### Slake
    #parser.add_argument("--question-file", type=str, default="/media/pc/d/2025/BigData/Med_MEGA/finetune_images/medvqa/Test_MedVQA_summary_Final.json")
    ##### PathVQA
    parser.add_argument("--question-file", type=str,default="/media/pc/d/2025/BigData/Med_MEGA/finetune_images/PathVQA/Test_PathVQA_summary_Final.json")

    parser.add_argument("--answers-file", type=str, default="/media/pc/d/2025/BigData/Med_MEGA/finetune_images/PathVQA/TestPathVQA_Predictions_Model_QueenVL_ZeroShot.json")


    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    eval_model(args)


