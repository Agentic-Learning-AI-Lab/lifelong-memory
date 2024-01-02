import pandas as pd
import traceback
import io
import argparse
from openai import AzureOpenAI
from openai import OpenAI
from tqdm import tqdm
from utils import load_annotations, load_clip_start_end_frame, load_video_frames
from prompts import load_template

def call_openai_api(client, system_prompt, user_prompt, model_name):
    prompt_messages = [ {"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    params = {
        "messages": prompt_messages,
        "temperature": 0,
        "max_tokens": 4096,
        "model": model_name
    }
    return client.chat.completions.create(**params)


def llm_inference(client, model_name, template, queries, clip_time_dic, captions=None, video_path=None):
    response_df = pd.DataFrame()
    for vid, cid in tqdm(list(queries.groups)):
        try:
            # load queries
            clip_query_df = queries.get_group((vid,cid))
            clip_query_text = clip_query_df[['query_index', 'query']].to_string(index=False)
            # load captions or frames
            if captions:
                clip_caption_df = captions.get_group((vid,cid)) 
                clip_memory = clip_caption_df[['timestamp', 'caption']].to_string(index=False)
            else:
                clip_memory = load_video_frames(video_path, vid, *clip_time_dic[(vid,cid)])
            # get llm predictions
            system_prompt = template.get_system_prompt()
            user_prompt = template.get_user_prompt(clip_query_text, clip_memory) 
            response = call_openai_api(client, system_prompt, user_prompt, model_name)
            data = response.choices[0].message.content
            result_df = pd.read_csv(io.StringIO(data), sep='\t')
            response_df = pd.concat([response_df, result_df], ignore_index=True)
        except:
            print(f"ERROR: {vid}, {cid}", flush=True)
            traceback.print_exc()
    return response_df

def main(args):
    # initialize openai
    if args.azure:
        client = AzureOpenAI(api_key=args.openai_key, api_version="2023-05-15", azure_endpoint=args.openai_endpoint)
    elif 'vicuna' in args.openai_model:
        client = OpenAI(api_key=args.openai_key, base_url=args.openai_endpoint)
    else:
        client = OpenAI(api_key=args.openai_key)  
    template = load_template(args.template)
    queries = load_annotations(args.annotation_path).groupby(['vid','cid'])
    clip_time_dic = load_clip_start_end_frame(args.annotation_path)
    cap_df = pd.read_csv(args.caption_path).groupby(['vid','cid']) if args.caption_path else None
    response_df = llm_inference(client, args.openai_model, template, queries, clip_time_dic, cap_df, args.video_path)
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
    args = parser.parse_args()
    main(args)

