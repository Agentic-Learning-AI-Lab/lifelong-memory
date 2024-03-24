import pandas as pd
import traceback
import io
import argparse
import time
import openai
from openai import AzureOpenAI
from openai import OpenAI
from tqdm import tqdm
from utils import load_queries, load_clip_start_end_frame, load_video_frames
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


def llm_inference(client, model_name, template, queries, captions=None, clip_time_dic=None, video_path=None):
    responses = []
    for k, query_text in tqdm(list(queries.items())):
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
                response = call_openai_api(client, system_prompt, user_prompt, model_name)
                data = response.choices[0].message.content
                if template.output_format == 'tsv':
                    data = pd.read_csv(io.StringIO(data), sep='\t').to_dict('records')
                    responses += data
                else:
                    data = eval(data)
                    data["q_uid"] = k
                    responses.append(data)
            except openai.RateLimitError:
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
        client = AzureOpenAI(api_key=args.openai_key, api_version="2023-05-15", azure_endpoint=args.openai_endpoint)
    elif 'vicuna' in args.openai_model:
        client = OpenAI(api_key=args.openai_key, base_url=args.openai_endpoint)
    else:
        client = OpenAI(api_key=args.openai_key)  
    template = load_template(args.task)
    queries = load_queries(args.annotation_path, args.task)
    clip_time_dic = None
    if 'vision' in args.task and 'nlq' in args.task:
        clip_time_dic = load_clip_start_end_frame(args.annotation_path)
    memory = pd.read_csv(args.caption_path).groupby(template.search_key) if args.caption_path else None
    response_df = llm_inference(client, args.openai_model, template, queries, memory, clip_time_dic, args.video_path)
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
    parser.add_argument('--task', type=str, help='Name of task, should contain NLQ or QA.', default = 'QA')
    args = parser.parse_args()
    main(args)

