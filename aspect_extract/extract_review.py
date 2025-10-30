import os
import json
import re
from tqdm import tqdm
import threading
import tempfile
import shutil
import GPUtil
from bs4 import BeautifulSoup
import hashlib  
from transformers import AutoTokenizer
import torch
from collections import Counter, defaultdict
from vllm import LLM, SamplingParams
from huggingface_hub import login

from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
from datetime import datetime, timezone


logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

A_STAR = [
    "ease of use", "sound quality", "value for money",
    "durability", "compatibility", "design", "customer support",
    "portability", "performance"
]
A_STAR_SET = {a.lower() for a in A_STAR}


class HistorySets:
    def __init__(self):
        self.user_hist: Dict[str, set] = defaultdict(set)  # user -> set[str]
        self.item_hist: Dict[str, set] = defaultdict(set)  # item -> set[str]

    def get_ui_union(self, u: str, i: str) -> List[str]:
        return sorted(list(self.user_hist.get(u, set()) | self.item_hist.get(i, set())))

    def update(self, u: str, i: str, features: List[str]):
        legal = [f.lower() for f in features if isinstance(f, str) and f.lower() in A_STAR_SET]
        if not legal:
            return
        self.user_hist[u].update(legal)
        self.item_hist[i].update(legal)


def to_datetime_utc(value):
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:  # 认为是毫秒
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return to_datetime_utc(int(s))
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                dt = datetime.strptime(s, fmt)
                if "%z" not in fmt:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
    return None

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
gpus = GPUtil.getAvailable(order="first", limit=4)  
# gpus = [4, 5, 6, 7]  #
if not gpus:
    raise RuntimeError("未找到可用的 GPU，请检查系统配置。")
gpus_str = ",".join(map(str, gpus))

os.environ["CUDA_VISIBLE_DEVICES"] = gpus_str

login(token='')
class SubTerm(BaseModel):
    feature: str
    opinion: Optional[str] = None
    sentence: Optional[str] = None
    sentiment_score: Optional[int] = None

class Aspect(BaseModel):
    aspects: List[SubTerm]

def init_local_llama_model(model_name='', model_path='', batch_size=16, max_num_batched_tokens=1024):
    llm = LLM(
        model=model_name,
        tensor_parallel_size=len(gpus),
        trust_remote_code=True,
        dtype="float16",  
        gpu_memory_utilization=0.95,  
        max_num_seqs=batch_size,  
        max_num_batched_tokens=max_num_batched_tokens,  
        enforce_eager=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path, use_fast=False)
    return tokenizer, llm

def clean_review_text(text: str) -> str:
    soup = BeautifulSoup(text or "", "html.parser")
    clean_text = soup.get_text(separator=" ").strip()
    return clean_text

def clean_generated_text(generated_text: str) -> Optional[str]:
    try:
        json_objects = re.findall(r'\{[^{}]*\}', generated_text)
        if not json_objects:
            logging.error("未找到任何 JSON 对象。")
            return None
        limited_json_objects = json_objects[:4]
        cleaned_text = "[" + ",".join(limited_json_objects) + "]"
        try:
            json.loads(cleaned_text)
            return cleaned_text
        except json.JSONDecodeError as e:
            logging.error(f"组合后的 JSON 无法解析: {e}")
            return None
    except Exception as e:
        logging.error(f"清洗生成文本时发生异常: {e}")
        return None

def parse_aspects(generated_text: str) -> List:
    cleaned_text = clean_generated_text(generated_text)
    if not cleaned_text:
        logging.error("清洗后的文本为空，无法解析。")
        return []

    try:
        aspects = json.loads(cleaned_text)
        sentences = []
        for aspect in aspects:
            feature = str(aspect.get("feature", "") or "").strip()
            opinion = str(aspect.get("opinion", "") or "").strip()
            sentiment_score = aspect.get("sentiment_score", 0)
            if sentiment_score is None:
                sentiment_score = 0

            if isinstance(sentiment_score, list):
                sentiment_score = sentiment_score[0] if len(sentiment_score) > 0 else 0
            try:
                sentiment = int(sentiment_score)
            except ValueError:
                sentiment = 0

            if feature and opinion:
                sentences.append([feature, opinion, sentiment])
        return sentences
    except json.JSONDecodeError as e:
        print(f"JSON 解析错误: {e}")
        return []

def format_output(data, parsed_sentences):
    return {
        "user": data.get("user_id", ""),
        "item": data.get("asin", ""),
        "rating": int(data.get("rating", 0)),
        "text": data.get("text", ""),
        "datetime": data.get("datetime", ""),
        "sentence": parsed_sentences,
    }

def build_prompt_global(a_star: List[str]) -> str:
    listing = ", ".join(a_star)
    return f"""
ROLE: 
You are an information extractor.

TASK: 
Please extract **less than 4** feature-opinion pairs from the following review and ensure that the features are **restricted to the following themes**:
[{listing}]
Output them strictly in the following JSON format:
```json
[
{{"feature": "Feature1", "opinion": "Opinion1", "sentiment_score": 1}},
{{"feature": "Feature2", "opinion": "Opinion2", "sentiment_score": -1}},
{{"feature": "Feature3", "opinion": "Opinion3", "sentiment_score": 0}}
]
```

**HARD RULES:**
- The opinion field must be extracted directly from the original review text and **must not be fabricated or paraphrased**.
- **Output only JSON data, do not include any text or explanation outside the JSON format.**
- Each feature-opinion pair must contain three fields "feature", "opinion" and "sentiment_score".
- Do not extract features that are not mentioned in the text.
- If there is no feature-opinion pair in the review, please output an empty JSON array `[]`.
- **Make sure the JSON array ends with ] and there are no syntax errors. **    
"""

def build_prompt_personal(H_ui_features: List[str]) -> str:
    if not H_ui_features:
        return "\nIf you are unsure which aspects to focus on when extracting feature–opinion pairs from the review, you may refer to Personal prior aspects []. \nNote: PERSONAL PRIOR is OPTIONAL guidance and MUST NOT override HARD RULES.\n"
    # 展示历史集合（名称形式），不排序、不截断
    listing = ", ".join(H_ui_features)
    return f"\nIf you are unsure which aspects to focus on when extracting feature–opinion pairs from the review, you may refer to Personal prior aspects[{listing}].\nNote: PERSONAL PRIOR is OPTIONAL guidance and MUST NOT override HARD RULES.\n"

def build_prompt_extract(review_text: str) -> str:
    return f"""
Now, please process the following review text:
{review_text}
"""
def build_dynamic_prompt(a_star: List[str], H_ui_features: List[str], review_text: str) -> str:
    # P_dynamic = [P_global[A*]; P_personal[H_ui(τ)]; P_extract[r]]
    return (
        build_prompt_global(a_star)
        + build_prompt_personal(H_ui_features)
        + build_prompt_extract(review_text)
    )

def dedup_triples_exact(triples):
    seen = set()
    out = []
    for t in triples:
        key = tuple(t)          # [feature, opinion, sentiment]
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out

def process_review_batch_dynamic(batch_reviews: List[dict], tokenizer, llm, sampling_params, hist: HistorySets):
    prompts = []
    for review in batch_reviews:
        clean_text = clean_review_text(review.get('text', ''))
        u = str(review.get("user_id", ""))
        i = str(review.get("asin", ""))
        H_ui = hist.get_ui_union(u, i) 
        prompt = build_dynamic_prompt(A_STAR, H_ui, clean_text)
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)

    parsed_sentences_list = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        # print(f"Generated Text: {generated_text}")  
        sentences = parse_aspects(generated_text)
        sentences = dedup_triples_exact(sentences)
        filtered = []
        for feat, op, sent in sentences:
            if feat.lower() in A_STAR_SET:
                filtered.append([feat, op, int(-1 if sent < 0 else 1 if sent > 0 else 0)])
        parsed_sentences_list.append(filtered[:4])  # 最多 4 条
    return parsed_sentences_list

def count_lines(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def load_processed_entries(output_path):
    if not os.path.exists(output_path):
        return Counter()
    processed_counter = Counter()
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                key = (
                    entry.get("rating"),
                    entry.get("text"),
                    entry.get("item"),
                    entry.get("user"),
                    entry.get("datetime")
                )
                processed_counter[key] += 1
            except json.JSONDecodeError:
                continue
    return processed_counter

def generate_hash(entry: dict) -> str:
    entry_str = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(entry_str.encode('utf-8')).hexdigest()

def load_jsonl(file_path, field_mapping):
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在。")
        return []
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            try:
                entry = json.loads(line.strip())
                rating = entry.get(field_mapping.get("rating", "rating"))
                text = entry.get(field_mapping.get("text", "text"))
                asin = entry.get(field_mapping.get("asin", "asin"))
                user_id = entry.get(field_mapping.get("user_id", "user_id"))
                datetime_v = entry.get(field_mapping.get("datetime", "datetime"))
                if all([rating is not None, text, asin, user_id, datetime_v]):
                    data.append((rating, text, asin, user_id, datetime_v))
                else:
                    print(f"警告: 文件 {file_path} 的第 {line_number} 行缺少必要字段，已跳过。")
            except json.JSONDecodeError:
                print(f"警告: 文件 {file_path} 的第 {line_number} 行不是有效的 JSON，已跳过。")
                continue
    return data

def count_matching_entries(file1_data, file2_data):
    set1 = set(file1_data)
    set2 = set(file2_data)
    matching = set1.intersection(set2)
    return len(matching)

def remove_duplicate_entries(file_path, matching_entries, field_mapping):
    seen = set()
    total_matching = 0
    temp_fd, temp_path = tempfile.mkstemp()
    with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file, open(file_path, 'r', encoding='utf-8') as original_file:
        for line_number, line in enumerate(original_file, 1):
            try:
                entry = json.loads(line.strip())
                rating = entry.get(field_mapping.get("rating", "rating"))
                text = entry.get(field_mapping.get("text", "text"))
                asin = entry.get(field_mapping.get("asin", "asin"))
                user_id = entry.get(field_mapping.get("user_id", "user_id"))
                datetime_v = entry.get(field_mapping.get("datetime", "datetime"))
                current_entry = (rating, text, asin, user_id, datetime_v)
                if current_entry in matching_entries:
                    if current_entry not in seen:
                        temp_file.write(line)
                        seen.add(current_entry)
                        total_matching += 1
                    else:
                        pass
                else:
                    pass
            except json.JSONDecodeError:
                print(f"警告: 文件 {file_path} 的第 {line_number} 行不是有效的 JSON，已跳过。")
                continue
    shutil.move(temp_path, file_path)
    return total_matching


def process_reviews(review_list: List[dict], processed_counter: Counter, tokenizer, llm, sampling_params, output_path: str, batch_size: int = 16):
    to_process = []
    for review in review_list:
        key = (
            review["rating"],
            review["text"],
            review["asin"],
            review["user_id"],
            review["datetime"]
        )
        if processed_counter[key] > 0:
            processed_counter[key] -= 1
            continue
        else:
            to_process.append(review)

    to_process.sort(key=lambda r: to_datetime_utc(r.get("datetime")) or datetime.now(timezone.utc))

    print(f"需要处理的条目数量: {len(to_process)}")

    hist = HistorySets()  
    with open(output_path, 'a', encoding='utf-8') as f_out:
        for idx in tqdm(range(0, len(to_process), batch_size), desc="Processing reviews", total=(len(to_process) + batch_size - 1) // batch_size):
            batch_reviews = to_process[idx: idx + batch_size]

            parsed_sentences_list = process_review_batch_dynamic(batch_reviews, tokenizer, llm, sampling_params, hist)

            for review, sentences in zip(batch_reviews, parsed_sentences_list):
                output = format_output(review, sentences)
                f_out.write(json.dumps(output, ensure_ascii=False) + '\n')

                features_in_this_review = [s[0] for s in sentences if isinstance(s, list) and len(s) >= 1]
                hist.update(str(review["user_id"]), str(review["asin"]), features_in_this_review)

if __name__ == "__main__":
    model_name = ''  
    model_path = '' 
    batch_size = 64  
    max_num_batched_tokens = 2048

    tokenizer, llm = init_local_llama_model(model_name, model_path, batch_size=batch_size, max_num_batched_tokens=max_num_batched_tokens)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=200
    )

    data_path = '/home/zheng/filtered_Musical_Instruments.jsonl'
    output_path = '/home/zheng/filtered_Musical_Instruments_out.jsonl'

    print("正在加载已处理的条目...")
    processed_counter = load_processed_entries(output_path)
    total_processed = sum(processed_counter.values())
    print(f"已处理的条目数: {len(processed_counter)}")

    print("正在加载输入文件并收集所有条目...")
    review_list = []
    skipped_due_to_missing_fields = 0
    skipped_due_to_invalid_json = 0
    total_entries = 0

    with open(data_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            total_entries += 1
            try:
                original_review = json.loads(line.strip())
                filtered_review = {
                    "rating": original_review.get("rating"),
                    "text": original_review.get("text", ""),
                    "asin": original_review.get("asin"),
                    "user_id": original_review.get("user_id"),
                    "datetime": original_review.get("datetime"), 
                }
                if all(filtered_review.values()):
                    review_list.append(filtered_review)
                else:
                    skipped_due_to_missing_fields += 1
                    missing_fields = [k for k, v in filtered_review.items() if not v]
                    logging.error(f"跳过缺少字段的条目，第 {line_number} 行，缺少字段: {missing_fields}，内容: {filtered_review}")
            except json.JSONDecodeError:
                skipped_due_to_invalid_json += 1
                logging.error(f"输入文件 {data_path} 的第 {line_number} 行不是有效的 JSON，已跳过。")
                continue

    print(f"总条目数: {total_entries}")
    print(f"有效条目数: {len(review_list)}")
    print(f"缺少字段的条目数: {skipped_due_to_missing_fields}")
    print(f"无效 JSON 的条目数: {skipped_due_to_invalid_json}")

    process_reviews(review_list, processed_counter, tokenizer, llm, sampling_params, output_path, batch_size=batch_size)

    output_line_count = count_lines(output_path)
    expected_count = total_entries - skipped_due_to_invalid_json - skipped_due_to_missing_fields
    print(f"输出文件的总条目数: {output_line_count}")
    print(f"预期的条目数: {expected_count}")
    if output_line_count == expected_count:
        print("验证成功：输出文件的条目数与输入文件的有效条目数相同。")
    else:
        remaining = expected_count - output_line_count
        print(f"验证失败：输出文件的条目数为 {output_line_count}，但预期为 {expected_count}，缺少 {remaining} 条目。")
