# LifelongMemory: Leveraging LLMs for Answering Queries in Long-form Egocentric Videos

**Abstract:** We introduce LifelongMemory, a new framework for accessing long-form egocentric videographic memory through natural language question answering and retrieval. LifelongMemory generates concise video activity descriptions of the camera wearer and leverages the reasoning and contextual understanding capabilities of pretrained large language models to produce precise answers. It further improves by using a confidence and refinement module to provide confident answers. Our approach achieves state-of-the-art performance on the EgoSchema benchmark for question answering and is highly competitive on the natural language query (NLQ) challenge of Ego4D.

<p align="center">
  &#151; <a href="https://lifelongmemory.github.io/"><b>View Paper Website</b></a> &#151;
</p>

<be>

![](https://github.com/Agentic-Learning-AI-Lab/lifelong-memory/blob/main/pipeline.png)

## Quick start

Captions (LaViLa on every 2s video clip + caption digest): [Google drive link](https://drive.google.com/file/d/1uNIcw0r3UnPoHQ4fJEqRHB2gUhQT4HWj/view?usp=sharing).
Note that the caption csv is assumed to have 4 columns: cid, vid, timestamp, caption for following steps.

#### LLM reasoning for Ego4D NLQ & EgoSchema (Video QA) 

If you are using OpenAI [available models](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
```
python scripts/llm_reason.py \
    --task QA (change to NLQ if your task is NLQ) \
    --annotation_path <the path to the official Ego4D QA/NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos, which contains 4 columns: cid, vid, timestamp, caption> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --llm_model <gpt model name, e.g. gpt-4o> \
    --api_key <your OpenAI API key> 
```

If you are using OpenAI via Azure
```
python scripts/llm_reason.py \
    --task QA (change to NLQ if your task is NLQ) \
    --annotation_path <the path to the official Ego4D QA/NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --azure \
    --endpoint  <your OpenAI endpoint e.g. https://xxxxxxxx.openai.azure.com/> \
    --llm_model <gpt model name> \
    --api_key <your Azure OpenAI API key>
```

If you are using Anthropic [available models](https://docs.anthropic.com/en/docs/about-claude/models)
```
python scripts/llm_reason.py \
    --task QA (change to NLQ if your task is NLQ) \
    --annotation_path <the path to the official Ego4D QA/NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --anthropic  \
    --llm_model <claude model name, e.g. claude-3-5-sonnet-20240620> \
    --api_key <your anthropic API key>
```

If you are using Llama, check its documentation on [Meta Llama](https://github.com/meta-llama/llama-models).
```
python scripts/llm_reason.py \
    --task QA (change to NLQ if your task is NLQ) \
    --annotation_path <the path to the official Ego4D QA/NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --llm_model <local path to llama model, e.g. /vast/work/public/ml-datasets/llama-3/Meta-Llama-3-8B-Instruct>
```

If you are using Vicuna, check its documentation on [OpenAI-Compatible RESTful APIs](https://github.com/lm-sys/FastChat?tab=readme-ov-file#openai-compatible-restful-apis--sdk)
```
python scripts/llm_reason.py \
    --task QA (change to NLQ if your task is NLQ) \
    --annotation_path <the path to the official Ego4D QA/NLQ annotation file> \
    --caption_path  <the path to the csv containing captions from the egocentric videos> \
    --output_path <the path to the csv containing the responses from the LLM> \
    --endpoint http://localhost:8000/v1 \
    --llm_model vicuna-7b-v1.5 
```

#### Caption Digest (optional)

This implementation uses [LaViLa pretrained textual encoder](https://github.com/facebookresearch/LaViLa/tree/main) to calculate similarity between texts. You may change the encoder to other encoders, such as the textual encoder of CLIP. 
Note that the LLM is only used for merging here. Since this is an easy task, consider using smaller models (such as gpt3.5, llama-3-8b) to save costs.
```
python scripts/caption_digest.py \
    --annotation_path <the path to the official Ego4D QA/NLQ annotation file> \
    --caption_path  <the path to the csv containing raw captions from the egocentric videos> \
    --output_path <the path to the csv containing the preprocessed captions> \
    --llm_model <gpt model name, e.g. gpt-3.5-turbo> \
    --api_key <your OpenAI API key> \
    --lavila_ckp_path <local path that contains LaViLa checkpoints, e.g. LaViLa/modelzoo/> \
    --alpha <1/2 of the captioning interval in frames. The default value is 30 because we generate captions every 2s(=60frames). You may need to adjust this value>
```
