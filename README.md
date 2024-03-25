# LifelongMemory: Leveraging LLMs for Answering Queries in Long-form Egocentric Videos

**Abstract:** We introduce LifelongMemory, a new framework for accessing long-form egocentric videographic memory through natural language question answering and retrieval. LifelongMemory generates concise video activity descriptions of the camera wearer and leverages the reasoning and contextual understanding capabilities of pretrained large language models to produce precise answers. It further improves by using a confidence and refinement module to provide confident answers. Our approach achieves state-of-the-art performance on the EgoSchema benchmark for question answering and is highly competitive on the natural language query (NLQ) challenge of Ego4D.

<p align="center">
  &#151; <a href="https://lifelongmemory.github.io/"><b>View Paper Website</b></a> &#151;
</p>

<be>

![](https://github.com/Agentic-Learning-AI-Lab/lifelong-memory/blob/main/pipeline.png)

### Quick start

#### Ego4D NLQ
```
python scripts/llm_reason.py \
    --task NLQ \
    --annotation_path <the path to the official NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos, which contains 4 columns: cid, vid, timestamp, caption> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --openai_model <gpt model name (check https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)> \
    --openai_key <your OpenAI API key> 
```

If you are using Azure
```
python scripts/llm_reason.py \
    --task NLQ \
    --annotation_path <the path to the official NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --azure \
    --openai_endpoint  <your OpenAI endpoint e.g. https://xxxxxxxx.openai.azure.com/> \
    --openai_model <gpt model name> \
    --openai_key <your Azure OpenAI API key>
```

If you are using Vicuna, check its documentation on [OpenAI-Compatible RESTful APIs](https://github.com/lm-sys/FastChat?tab=readme-ov-file#openai-compatible-restful-apis--sdk)
```
python scripts/llm_reason.py \
    --task NLQ \
    --annotation_path <the path to the official NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --openai_endpoint http://localhost:8000/v1 \
    --openai_model vicuna-7b-v1.5 
```

#### EgoSchema (Video QA)
```
python scripts/llm_reason.py \
    --task QA
    --annotation_path <the path to the official EgoSchema question file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos, which contains 4 columns: q_uid, timestamp, caption> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --openai_model <gpt model name (check https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)> \
    --openai_key <your OpenAI API key> \
```



