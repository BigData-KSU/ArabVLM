import json
import re
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.cider.cider_scorer import CiderScorer
#nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#nltk.download('wordnet')



smoothie = SmoothingFunction().method4
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
cider_scorer = CiderScorer(n=4, sigma=6.0)

def split_sentence(sentence, n):
    words = defaultdict(int)
    tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words

def calculate_exactmatch(candidate, reference):

    #print(candidate)
    #print(reference)
    candidate_words = split_sentence(str(candidate), 1)
    reference_words = split_sentence(str(reference), 1)

    #print(reference_words)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]

    if total == 0.:
        return 0 ###"0 (warning: length of candidate's words is 0)"
    else:
        return count / total

## PathVQA
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/PathVQA/PathVQA_Predictions_Model_videollava_ROCO_BLIP-04June.json'
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/PathVQA/TestPathVQA_Predictions_Model_QueenVL_ZeroShot.json'


### Slake
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/slake/Slake_Predictions_Model_videollava_ROCO_BLIP-04June.json'
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/slake/Slake_Predictions_Model_videollava_ROCO_BLIP-04June-Zero-Shot.json'
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/slake/TestSlake_Predictions_Model_videollava_06Aug.json'
### VQA2019
file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/VQA2019/Vmed2019_Predictions_Model_videollava_ROCO_BLIP-04June.json'
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/VQA2019/Vmed2019_Predictions_Model_videollava_ROCO_BLIP-04June-ZeroShot.json'

#### medVQA
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/medvqa/TestMedVQA_Predictions_Model_videollava_ROCO_BLIP-04June.json'
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/medvqa/TestMedVQA_Predictions_Model_videollava_ROCO_BLIP-04June-ZeroShot.json'
#file='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/medvqa/TestMedVQA_Predictions_Model_videollava.json'
# Load the JSON file
with open(file, 'r', encoding='utf-8') as file:
    data = json.load(file)



###### read VQA2019 File
import json
# Function to
# read and process the text file

filename='/media/pc/d/2025/BigData/Med_MEGA/finetune_images/VQA2019/VQAMed2019_Test_Questions_w_Ref_Answers.txt'
# Function to read and process the text file
def read_and_process_file(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            # Remove any leading/trailing whitespace characters (e.g., newline characters)
            line = line.strip()
            # Split the line into components based on the '|' delimiter
            parts = line.split('|')
            if len(parts) == 4:
                image, modality, question, answer = parts
                # Store the components in a dictionary for clarity
                data.append({
                    'image': image,
                    'modality': modality,
                    'question': question,
                    'answer': answer
                })
    return data

# Example usage
data_new = read_and_process_file(filename)




correct_yes_no=0
Total_yes_no=0
Total_open_1=0
Total_open_2=0
Total_open_3=0
Total_open_4=0
OA_open=0
total_bleu1=0
total_bleu2=0
total_bleu3=0
total_bleu4=0
tot_rouge=0
tot_cider=0
for item2 in data:
    item3=item2['answer']
    item4=item2['pred_answer']
    for idx in range(len(item3)):
        if item3[idx].lower()=='yes' or item3[idx].lower()=='no':
            Total_yes_no=Total_yes_no+1
            if item3[idx].lower()==item4[idx].lower().split('.')[0]:
               correct_yes_no=correct_yes_no+1

        if item3[idx].lower()!= 'yes' or item3[idx].lower() != 'no':
            cap_result = item4[idx].lower().translate({ord(i): None for i in '..'}).split()
            print(cap_result)
            ref_sentence = item3[idx].lower().translate({ord(i): None for i in '..'}).lower().split()
            print(ref_sentence)

            if data_new['']
            Total_open = Total_open + 1

            # print(cap_result)
            # print(ref_sentence)
            OA_open = OA_open + calculate_exactmatch(cap_result, ref_sentence)

            # F1_score_res = F1_score_res + calculate_f1score(cap_result, ref_sentence)
            ref_sentence = item2['answer'][0].lower().split('.')[0]
            cap_result = item2['pred_answer'][0].lower().split('.')[0]
            # BLEU scores
            total_bleu1 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(1, 0, 0, 0), smoothing_function=smoothie)
            total_bleu2 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(0.5, 0.5, 0, 0),smoothing_function=smoothie)
            total_bleu3 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
            total_bleu4 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smoothie)



OA_yes=100*correct_yes_no/Total_yes_no
print('OA yes no:',100*correct_yes_no/Total_yes_no)
print('Accuracy OpenEnded:', 100 * OA_open / Total_open)
print('Accuracy All :', 100 * (OA_open + correct_yes_no) / (Total_yes_no + Total_open))

print('BLeu1: ',100*total_bleu1/Total_open)
print('BLeu2: ',100*total_bleu2/Total_open)
print('BLeu3: ',100*total_bleu3/Total_open)
print('BLeu4: ',100*total_bleu4/Total_open)

print('++++++++++++++++++++++++++++++++++++++++++++++++')
print('Caption Results Task..............................')

total_bleu1=0
total_bleu2=0
total_bleu3=0
total_bleu4=0
tot_rouge=0
tot_cider=0
tot_meteor=0
for item2 in data:
    ref_sentence=item2['summary'][0].lower().split('.')[0]
    cap_result=item2['summary_pred'][0].lower().split('.')[0]

    total_bleu1 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(1, 0, 0, 0),  smoothing_function=smoothie)
    total_bleu2 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(0.5, 0.5, 0, 0),smoothing_function=smoothie)
    total_bleu3 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(0.33, 0.33, 0.33, 0),smoothing_function=smoothie)
    total_bleu4 += sentence_bleu([ref_sentence.split()], cap_result.split(), weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smoothie)
    # ROUGE score
    rouge_scores = scorer.score(' '.join(ref_sentence), ' '.join(cap_result))
    tot_rouge += rouge_scores['rouge1'].fmeasure

    # CIDEr score
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([cap_result] + [ref_sentence])
    # Compute cosine similarities
    cos_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    # CIDEr score is the average cosine similarity
    cider_score = np.mean(cos_similarities)
    tot_cider += cider_score

    # Compute METEOR score for each reference
    # Tokenize the sentences
    #hypothesis_tokens = nltk.word_tokenize(cap_result)
    #reference_tokens = nltk.word_tokenize(ref_sentence)
    # Calculate METEOR score
    #score = meteor_score([reference_tokens], hypothesis_tokens)
    #tot_meteor=tot_meteor+score

print('BLeu1: ',100*total_bleu1/len(data))
print('BLeu2: ',100*total_bleu2/len(data))
print('BLeu3: ',100*total_bleu3/len(data))
print('BLeu4: ',100*total_bleu4/len(data))
print('ROUGE: ',100*tot_rouge/len(data))
print('Cider: ',100*tot_cider/len(data))



