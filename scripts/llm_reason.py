import pandas as pd
import traceback
import io
import argparse
import time
import openai
from openai import AzureOpenAI
from openai import OpenAI
from tqdm import tqdm
from utils import load_queries, load_clip_start_end_frame, load_video_frames, postprocess
from prompts import load_template
import os
import sys
import json
import anthropic
import re

def call_llm_api(client, system_prompt, user_prompt, model_name, max_len=4096, temp=0):
    if "llama" in model_name.lower():
        from models.llama3.api.datatypes import SystemMessage, UserMessage
        prompt_messages = [SystemMessage(content=system_prompt), UserMessage(content=user_prompt)]
        response = client.chat_completion(
            prompt_messages,
            temperature=temp,
            max_gen_len=max_len,
        )
        return response.generation.content
    prompt_messages = [ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if 'claude' in model_name.lower():
        response = client.messages.create(
            system = system_prompt,
            messages = [prompt_messages[1]],
            temperature=temp,
            max_tokens=max_len,
            model=model_name,
        )
        return response.content[0].text
    else:
        response = client.chat.completions.create(
            messages = prompt_messages,
            temperature=temp,
            max_tokens=max_len,
            model=model_name,
        )
        return response.choices[0].message.content



def llm_inference(client, model_name, template, queries, captions=None, clip_time_dic=None, video_path=None):
    responses = []
    for k, query_text in tqdm(list(queries.items())[0:1]):
        # auto retry if reaching openai's rate limit
        for _ in range(3):
            try:
                # load queries
                clip_query_text = query_text
                # load captions if using GPT4, GPT3.5, Vicuna...
                if captions:
                    clip_caption_df = captions.get_group(k) 
                    clip_memory = clip_caption_df[['timestamp', 'caption']].to_string(index=False)
                # load frames if using GPT4V
                else:
                    vid = k[1] # Need the video_uid here
                    clip_memory = load_video_frames(video_path, vid, *clip_time_dic[k])
                # get llm predictions
                system_prompt = template.get_system_prompt()
                user_prompt = template.get_user_prompt(clip_query_text, clip_memory) 
                data = call_llm_api(client, system_prompt, user_prompt, model_name)
                if template.output_format == 'tsv':
                    formatted_data = pd.read_csv(io.StringIO(data.replace('tsv\n','').replace('```','')), sep='\t').to_dict('records')
                    responses += formatted_data
                if template.output_format == 'dic':
                    try:
                        formatted_data = eval(data)
                    except:
                        formatted_data = {}
                        formatted_data['prediction'] = postprocess(data)
                        formatted_data['explanation'] = data
                    formatted_data["q_uid"] = k
                    responses.append(formatted_data)
            except openai.RateLimitError or anthropic.RateLimitError:
                print("Too many request: Sleeping 1s", flush=True)
                time.sleep(1)
            except Exception:
                print(f"Error occurs when processing this query: {k}", flush=True)
                traceback.print_exc()
                break
            else:
                break
        else:
            print(f"Too many requests for this query: {k}", flush=True)
            traceback.print_exc()
    return pd.DataFrame(responses)

def main(args):
    if args.azure:
        client = AzureOpenAI(api_key=args.api_key, azure_endpoint=args.endpoint, api_version = '2023-05-15')
    elif 'vicuna' in args.llm_model:
        client = OpenAI(api_key=args.api_key, base_url=args.endpoint)
    elif 'llama' in args.llm_model:
        sys.path.append("llama-models") #TODO: Replace by your path to llama
        from models.llama3.reference_impl.generation import Llama
        llama_dir = args.llm_model   # "/vast/work/public/ml-datasets/llama-3/Meta-Llama-3-8B-Instruct"
        client = Llama.build(ckpt_dir=llama_dir, tokenizer_path=f"{llama_dir}/tokenizer.model", max_seq_len=4096, max_batch_size=1)
    elif args.anthropic:
        client = anthropic.Anthropic(api_key=args.api_key)
    else:
        client = OpenAI(api_key=args.api_key)  
    template = load_template(args.task)
    queries = load_queries(args.annotation_path, args.task)
    clip_time_dic = None
    if 'vision' in args.task and 'nlq' in args.task:
        clip_time_dic = load_clip_start_end_frame(args.annotation_path)
    memory = pd.read_csv(args.caption_path).groupby(template.search_key) if args.caption_path else None
    response_df = llm_inference(client, args.llm_model, template, queries, memory, clip_time_dic, args.video_path)
    response_df.to_csv(args.output_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LLM inner speech argument parser')
    parser.add_argument('--annotation_path', required=True, type=str, help='path to the ego4d annotation file')
    parser.add_argument('--caption_path', type=str, help='path to the captions from the egocentric video')
    parser.add_argument('--video_path', type=str, help='path to the raw egocentric videos')
    parser.add_argument('--output_path', required=True, type=str, help='path to the csv containing the results from GPT4')
    parser.add_argument('--api_key', type=str, help='openai/azure/anthropic api key', default = 'EMPTY')
    parser.add_argument('--llm_model', required=True, type=str, help='llm model name', default = 'gpt-3.5-turbo')
    parser.add_argument('--azure', action='store_true', help='is azure')
    parser.add_argument('--anthropic', action='store_true', help='is anthropic')
    parser.add_argument('--endpoint', type=str, help='end point (required by azure and vicuna)')
    parser.add_argument('--task', type=str, help='Name of task, should contain NLQ or QA.', default = 'QA')
    args = parser.parse_args()
    main(args)

