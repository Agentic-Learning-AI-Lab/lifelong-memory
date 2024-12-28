import json
import cv2
import pandas as pd
import base64
import numpy as np
import re

def extractImages(path, timestamp):
    vidcap = cv2.VideoCapture(path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, timestamp)    
    success, image = vidcap.read()   
    return success, image

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
        if 'videos' in data: data = data['videos'] #ego4d nlq
    return data

def load_video_frames(video_path, vid, start, end, freq=30*15):
    video = cv2.VideoCapture(f"{video_path}/{vid}.mp4")
    base64Frames = []
    cache = np.zeros((336,336))
    while start <= end:
        print(start, end)
        video.set(cv2.CAP_PROP_POS_FRAMES, start)  
        success, frame = video.read()  
        _, buffer = cv2.imencode(".jpg", frame)
        if success: cache = buffer
        base64Frames.append({"image": base64.b64encode(cache).decode("utf-8"), "resize": 336}) 
        start += freq
    video.release()
    return base64Frames

def load_queries(ann_path, task, start = 0, end = 3):
    annotations = load_json(ann_path)
    start = max(0, start)
    if end: end = min(end, len(annotations))
    if 'nlq' in task.lower():
        return load_ego4dnlq_queries(annotations[start:end])
    if 'qa' in task.lower():
        return load_egoschema_queries(annotations[start:end])

def load_ego4dnlq_queries(annotations):
    query_map = {}
    count = 0
    for v in annotations:
        vid = v['video_uid']
        for c in v['clips']:
            cid = c['clip_uid']
            for a in c['annotations']:
                clip_queries = f"query \t query_index"
                for query in a['language_queries']:
                    count += 1
                    if 'query' in query and query['query']:
                        clip_queries += f"{query['query']} \t {count} \n"
            query_map[(cid, vid)] = clip_queries
    return query_map

def load_egoschema_queries(annotations):
    query_map = {}
    for query in annotations:
        query_map[query['q_uid']] = f'Question: {query["question"]} \n' + '\n'.join([f"Option {i}: " + query[f"option {i}"] for i in range(5)])
    return query_map

def load_clip_start_end_frame(ann_path):
    clip_dic = {}
    annotations = load_json(ann_path)
    for v in annotations['videos']:
        vid = v['video_uid']
        for c in v['clips']:
            cid = c['clip_uid']
            clip_dic[(vid,cid)] = round(c['video_start_sec'] * 30), round(c['video_end_sec'] * 30)
    return clip_dic

PATTERN = re.compile(r'prediction\s*[^\d]*\s*(\d+)', re.IGNORECASE)
def postprocess(s):
    match = PATTERN.search(s)
    if match:
        return match.group(1)
    else:
        return re.findall('[0-9]+', s)[0]



