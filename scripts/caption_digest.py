import pandas as pd
import traceback
from tqdm import tqdm
import traceback
import io
import argparse
from openai import AzureOpenAI
from openai import OpenAI
import torch
import sys
import json
import os.path as osp
import certifi
import os
from collections import OrderedDict
from utils import load_json
from prompts import load_template
from llm_reason import call_llm_api
sys.path.append('LaViLa') #TODO: Replace by your path to LaViLa
from lavila.models.utils import inflate_positional_embeds
from lavila.utils.preprocess import generate_label_map, generate_tokenizer
from lavila.utils import distributed as dist_utils
from lavila.models import models


class LaViLaSimilarity:
    """
    Goal: Compute the embeddings of captions and compute similarity scores
    This implementation adopts LaViLa textual encoder. You may change the encoder to other encoders s.t. CLIP.
    Code adapted from LaViLa: https://github.com/facebookresearch/LaViLa/tree/main
    """
    def __init__(self, ckpt_path, ckpt_name='clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = osp.join(ckpt_path, ckpt_name)
        ckpt = torch.load(ckpt_path)
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        old_args = ckpt['args']
        print('=> creating model: {}'.format(old_args.model))
        self.encoder_tokenizer = generate_tokenizer(old_args.model)
        self.encoder = getattr(models, old_args.model)(
                text_use_cls_token=old_args.use_cls_token,
                project_embed_dim=old_args.project_embed_dim,
                gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
                timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
                timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
                freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
                freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
                num_frames=4,
                drop_path_rate=0,
            )
        if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
            print('=> inflating PE in models due to different frame numbers')
            state_dict = inflate_positional_embeds(
                    self.encoder.state_dict(), state_dict,
                    num_frames=4,
                    load_temporal_fix='bilinear',
                )
        self.encoder.load_state_dict(state_dict, strict=True)
        self.encoder.eval()
        self.encoder.to(device)
        torch.backends.cudnn.benchmark = True
        self.cos = torch.nn.CosineSimilarity()

    def get_text_features(self, caps):
        torch.backends.cudnn.benchmark = True
        with torch.no_grad():
            texts = self.encoder_tokenizer(caps).cuda(non_blocking=True)
            texts = texts.view(-1, 77).contiguous()
            text_feat = dist_utils.get_model(self.encoder).encode_text(texts)  
        return text_feat

    def get_similarity(self, s1, s2):
        return self.cos(self.get_text_features(s1),self. get_text_features(s2)).max().item()


AMBIGUOUS_EXPRESSION_LIST = ['looks around', 'walks around', 'look around', 'walk around']
def filter_caption(caption, question, filter_thresh = 0):
    caption = caption.replace('#C ', '').replace('#c ', '').replace('#O ', '').replace('#o ', '')
    for phrase in AMBIGUOUS_EXPRESSION_LIST:
        if phrase in caption: return None
    if filter_thresh > 0 and get_similarity(question, caption) < filter_thresh:
        return None
    return caption

def merge_captions(cap_list, client, model, prompt_template):
    if len(cap_list) == 0: return 
    if len(cap_list) == 1: return cap_list[0]
    user_prompt = prompt_template.get_user_prompt('\n'.join(cap_list))
    system_prompt = prompt_template.get_system_prompt()
    try:
        response = call_llm_api(client, system_prompt, user_prompt, model)
        response = response.choices[0].message.content.strip('"').strip("'")
        print('MERGE: ', cap_list, '->', response, flush=True)
        return response
    except:
        print('CANNOT MERGE: ', cap_list)
        traceback.print_exc()
        return cap_list[0]

def caption_digest(raw_caption_df, qa_dic, merge_fn, similarity_fn, alpha = 30, merge_thresh = 0.8):
    preprocessed_rows = []
    raw_caption_df = raw_caption_df.groupby(['q_uid'])
    for qid in tqdm(list(qa_dic.keys())):
        print(f'-------------------------{qid}-------------------------')
        clip_df = raw_caption_df.get_group(qid)
        question = qa_dic[qid]
        time_list = []
        cap_list = []
        for i, row in clip_df.iterrows():
            caption = filter_caption(row['caption'], question)
            if caption is None:
                print(f"Caption({caption}) not useful for answering question ({question}) -> skip")
                continue
            if len(cap_list)==0 or similarity_fn(cap_list[-1], caption) > merge_thresh:
                time_list.append(row['timestamp'])
                cap_list.append(caption)
            else:
                preprocessed_rows.append({'q_uid':qid, 'caption':  merge_fn(cap_list),  'timestamp': f'[{time_list[0]-alpha}, {time_list[-1]+alpha}]'})
                time_list = [row['timestamp']]
                cap_list = [caption]
        # the end of the clip: add the last caption
        preprocessed_rows.append({'q_uid':qid, 'caption': merge_fn(cap_list),  'timestamp': f'[{time_list[0]-alpha}, {time_list[-1]+alpha}]'})
    return pd.DataFrame(preprocessed_rows)


def main(args):
    if args.azure:
        client = AzureOpenAI(api_key=args.openai_key, api_version="2023-05-15", azure_endpoint=args.openai_endpoint)
    else:
        os.environ['SSL_CERT_FILE'] = certifi.where() # might need for openai api
        client = OpenAI(api_key=args.openai_key, base_url="https://api.openai.com/v1/") 
    qa_list = load_json(args.annotation_path)
    qa_dic = {v['q_uid']: v['question'] for v in qa_list}
    raw_caption_df = pd.read_csv(args.caption_path)
    prompt_template = load_template('merge')
    merge_fn = lambda x: merge_captions(x, client, args.openai_model, prompt_template)
    similarity_fn = LaViLaSimilarity(args.lavila_ckp_path).get_similarity
    preprocessed_df = caption_digest(raw_caption_df, qa_dic, merge_fn, similarity_fn, alpha=args.alpha)
    preprocessed_df.to_csv(args.output_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LifelongMemory caption digest argument parser')
    parser.add_argument('--annotation_path', required=True, type=str, help='path to the ego4d nlq annotation file')
    parser.add_argument('--lavila_ckp_path', required=True, type=str, help='Name of task, should contain NLQ or QA.', default = '/scratch/yw3076/LaViLa/modelzoo/')
    parser.add_argument('--caption_path', required=True, type=str, help='path to the captions from the egocentric video')
    parser.add_argument('--output_path', required=True, type=str, help='path to the csv containing the results from GPT4')
    parser.add_argument('--openai_key', required=True, type=str, help='openai api key', default = 'EMPTY')
    parser.add_argument('--openai_model', type=str, help='openai engine (name of deployment)', default = 'gpt-3.5-turbo')
    parser.add_argument('--alpha', type=int, help='1/2 of the captioning interval in frames', default = 30)
    parser.add_argument('--azure', action='store_true', help='is azure')
    parser.add_argument('--openai_endpoint', type=str, help='openai end point')
    args = parser.parse_args()
    main(args)
