import json
import cv2
import pandas as pd
import base64
import numpy as np

def extractImages(path, timestamp):
    vidcap = cv2.VideoCapture(path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, timestamp)    
    success, image = vidcap.read()   
    return success, image

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
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
    #print(len(base64Frames))
    return base64Frames

def load_annotations(ann_path, start = 0, end = None):
    query_rows = []
    count = 0
    annotations = load_json(ann_path)
    start = max(0, start)
    if end: end = min(end, len(annotations['videos']))
    for v in annotations['videos'][start:end]:
        vid = v['video_uid']
        for c in v['clips']:
            cid = c['clip_uid']
            for a in c['annotations']:
                for query in a['language_queries']:
                    count += 1
                    if 'query' in query and query['query']:
                        query_rows.append({'vid':vid, 'cid':cid, 'query': query['query'], 'query_index': count})
    query_df = pd.DataFrame(query_rows)
    return query_df

def load_clip_start_end_frame(ann_path):
    clip_dic = {}
    annotations = load_json(ann_path)
    for v in annotations['videos']:
        vid = v['video_uid']
        for c in v['clips']:
            cid = c['clip_uid']
            clip_dic[(vid,cid)] = round(c['video_start_sec'] * 30), round(c['video_end_sec'] * 30)
    return clip_dic



