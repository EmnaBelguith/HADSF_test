#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PROMPT = """
Please extract **less than 5** feature-opinion pairs from the following review and ensure that the features are **restricted to the following themes**: 
**service, food quality, ambiance, value for money, cleanliness, portion size, menu variety, overall experience, wait time**

Output them strictly in the following JSON format:
```json
[
{{"feature": "Feature1", "opinion": "Opinion1", "sentiment_score": 1}},
{{"feature": "Feature2", "opinion": "Opinion2", "sentiment_score": -1}},
{{"feature": "Feature3", "opinion": "Opinion3", "sentiment_score": 0}}
]
```
** Note:**
- **Output only JSON data, no other text description or code. **
- Each feature-opinion pair must contain three fields "feature", "opinion" and "sentiment_score".
- Do not extract features that are not mentioned in the text.
- The opinion field must be extracted directly from the original review text and **must not be fabricated or paraphrased**.
- If there is no feature-opinion pair in the review, please output an empty JSON array `[]`.
- **Make sure the JSON array ends with ] and there are no syntax errors. **

Example:

Review text:
```
The laptop is incredibly fast and lightweight. However, the battery life is too short for long hours of work, and the price feels a bit high for its features.
```

Output:
```json
[
{{"feature": "performance", "opinion": "incredibly fast", "sentiment_score": 1}},
{{"feature": "durability", "opinion": "battery life is too short", "sentiment_score": -1}},
{{"feature": "price", "opinion": "feels a bit high", "sentiment_score": 0}}
]
```

Now, please process the following review text:
This restaurant is great. I ordered the clam chowder which was excellent. I also ordered the muscles which were the best I've ever had. I would recommend eating at this restaurant. There is also an excellent view.
"""
def main():
    parser = argparse.ArgumentParser(description="Single-turn JSON extractor with vLLM")
    parser.add_argument("--model", type=str, default="/path/Llama-3.1-8B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--visible_devices", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--cache_dir", type=str, default="")
    args = parser.parse_args()

    if args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, use_fast=False)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=0.95,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate([PROMPT.strip()], sampling_params)
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()

"""
{"user": "joaoyZhKtmO5iNyoXSB2yQ", "item": "U_U6C9AF7xmQijTTUFRzkQ", "rating": 5, "text": "This restaurant is great. I ordered the clam chowder which was excellent. I also ordered the muscles which were the best I've ever had. I would recommend eating at this restaurant. There is also an excellent view.", "datetime": "2019-03-06 18:51:06", "sentence": [["service", "great", 1], ["food quality", "excellent", 1], ["food quality", "the best I've ever had", 1]]}
"""
