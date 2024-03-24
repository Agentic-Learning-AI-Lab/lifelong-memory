import pandas as pd
import traceback
import io
import argparse
import json
from openai import AzureOpenAI
from openai import OpenAI
from tqdm import tqdm
from utils import load_annotations, load_clip_start_end_frame, load_video_frames
from prompts import load_template
import time
import openai

sep = '\n\n###\n\n'
system_prompt = "You are presented with a textual description of a video clip. Your task is to answer a question related to this video, choosing the correct option out of five possible answers. It is crucial that you imagine the visual scene as vividly as possible to enhance the accuracy of your response. After selecting your answer, rate your confidence level in this choice on a scale from 1 to 3, where 1 indicates low confidence and 3 signifies high confidence. Please provide a concise one-sentence explanation for your chosen answer. If you are uncertain about the correct option, select the one that seems closest to being correct. "           
s2 = "The dictionary with keys of prediction, explanation, confidence, where prediction is a number. "

def call_openai_api(client, system_prompt, user_prompt, model_name):
    prompt_messages = [ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    params = {
        "messages": prompt_messages,
        "temperature": 0,
        "max_tokens": 1024,
        "model": model_name
    }
    return client.chat.completions.create(**params)


def llm_inference(client, model_name, grouped_df, questions_dic, idx=None):
    responses = []
    #df = pd.read_csv("/scratch/yw3076/LLM-Inner-Speech/results/qa_lavila_2s.csv")
    #done_list = set(df['q_uid'])
    #with open("/scratch/yw3076/egoschema-public/subset_answers.json", "r") as f:
    #    qa_subset = list(json.load(f).keys())
    q_list = list(questions_dic.keys())
    n = len(q_list)
    qa_subset = q_list[idx*250:min(n, (idx+1)*250)]
    for q in tqdm(qa_subset):
        query = questions_dic[q]
        #q = '00b9a0de-c59e-49cb-a127-6081e2fb8c8e'
        #query = questions_dic[q]
        #if q not in qa_subset: continue
        for attempt in range(3):
            try:
                captions = grouped_df.get_group(q)[['timestamp', 'caption']].to_string(index=False)
                user_prompt = prompt(captions, query)
                response = call_openai_api(client, system_prompt, user_prompt, model_name)
                data = response.choices[0].message.content
                print(q, data, flush=True)
                data = eval(data)
                data["q_uid"] = q
                responses.append(data)
            except openai.RateLimitError:
                print("Too many request: Sleeping 1s", flush=True)
                time.sleep(1)
            except Exception:
                print(f"----------------------------- ERROR: {q} ----------------------------------", flush=True)
                traceback.print_exc()
                break
            else:
                break
        else:
            print(f"----------------------------- ERROR: {q} ----------------------------------", flush=True)
            traceback.print_exc()
    return pd.DataFrame(responses)

def prompt(captions, query):
    return "Memory:" + sep + captions + sep + f'Questions: {query["question"]} \n' + '\n'.join([f"option {i}: " + query[f"option {i}"] for i in range(5)]) + sep + s2

def main(args):
    # initialize openai
    if args.azure:
        client = AzureOpenAI(api_key=args.openai_key, api_version="2023-05-15", azure_endpoint=args.openai_endpoint)
    elif 'vicuna' in args.openai_model:
        client = OpenAI(api_key=args.openai_key, base_url=args.openai_endpoint)
    else:
        client = OpenAI(api_key=args.openai_key)  
    
    with open(args.annotation_path) as f:
        questions = json.load(f)
    questions_dic = {}
    for question in questions:
        questions_dic[question['q_uid']] = question
        
    cap_df = pd.read_csv(args.caption_path)
    grouped_cap_df = cap_df.groupby(["q_uid"])
    
    response_df = llm_inference(client, args.openai_model, grouped_cap_df, questions_dic, args.idx)
    response_df.to_csv(args.output_path)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LLM inner speech argument parser')
    parser.add_argument('--annotation_path', required=True, type=str, help='path to the ego4d nlq annotation file')
    parser.add_argument('--caption_path', type=str, help='path to the captions from the egocentric video')
    parser.add_argument('--video_path', type=str, help='path to the raw egocentric videos')
    parser.add_argument('--output_path', required=True, type=str, help='path to the csv containing the results from GPT4')
    parser.add_argument('--openai_key', required=True, type=str, help='openai api key', default = 'EMPTY')
    parser.add_argument('--openai_model', required=True, type=str, help='openai engine (name of deployment)', default = 'gpt4-32k')
    parser.add_argument('--azure', action='store_true', help='is azure')
    parser.add_argument('--openai_endpoint', type=str, help='openai end point')
    parser.add_argument('--template', type=str, help='prompt template', default = 'default')
    parser.add_argument('--idx', type=int)
    args = parser.parse_args()
    main(args)

