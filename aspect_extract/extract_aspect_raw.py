import os
import json
import re
import random
from collections import Counter, defaultdict
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
import GPUtil

from transformers import AutoTokenizer
import torch
from vllm import LLM, SamplingParams
from huggingface_hub import login

# =========================
# 全局配置（可调）
# =========================
K = 1                          # 多重采样轮数（保留参数，这里按 item 流程不依赖 K）
consensus_tau_items = 5        # 共识阈值：某方面至少出现在 tau 个不同 item 中才保留

aspect_max_tokens = 256        # 单 item 抽方面的生成上限（一般不需要太长）
abs_max_tokens = 1024          # 摘要阶段生成上限
max_model_len = 4096
sample_fraction = 0.10         # 先对 item 下采样（0~1），生产/调试可改小以快速跑通
random_seed = 123

# —— 分桶阈值：判定“长 item”的总 token 数（按拼接后评论粗估）
LONG_ITEM_TOKENS = 6000

# 层次化摘要切块与合并
CHUNK_TOKEN_BUDGET = 2000
MERGE_TOKEN_BUDGET = 8192
RESERVE_TOKENS = 80
MAX_LEVELS = 5
MAX_CHUNKS_PER_ITEM = 64

# vLLM 外层批大小（一次性提交多少个 prompt）
MAX_PROMPTS_PER_CALL_ABS    = 256  # 摘要（短 item / 长 item 块级 / 合并级）外层批
MAX_PROMPTS_PER_CALL_MERGE  = 256  # 合并阶段外层批
MAX_PROMPTS_PER_CALL_ASPECT = 256  # 抽方面阶段外层批

SHOW_PROMPT_PROGRESS = True

# =========================
# 日志 & 登录
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 建议用环境变量：export HF_TOKEN=xxx；此处仅为示例
login(token=os.getenv("HF_TOKEN", "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"))

# =========================
# GPU
# =========================
gpus = GPUtil.getAvailable(limit=2)
if not gpus:
    raise RuntimeError("未找到可用的 GPU，请检查系统配置。")
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))

# =========================
# 模型
# =========================
def init_model(model_name, model_path, batch_size=64, max_model_len=4096):
    llm = LLM(
        model=model_name,
        tensor_parallel_size=len(gpus),
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        max_num_seqs=batch_size,    # 并发槽位上限（由 vLLM 控制）
        max_model_len=max_model_len,
        enforce_eager=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path, use_fast=False)
    return tokenizer, llm

# =========================
# 工具
# =========================
def clean_text(text: str) -> str:
    soup = BeautifulSoup(text or "", "html.parser")
    return soup.get_text(separator=" ").strip()

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

def truncate_to_fit(tokenizer, prefix: str, payload: str, budget_tokens: int) -> str:
    prefix_tokens = count_tokens(tokenizer, prefix)
    remain = max(0, budget_tokens - prefix_tokens - RESERVE_TOKENS)
    if remain <= 0:
        return ""
    toks = tokenizer.encode(payload, add_special_tokens=False)
    if len(toks) <= remain:
        return payload
    toks = toks[:remain]
    return tokenizer.decode(toks, skip_special_tokens=True)

def extract_json_array_block(s: str) -> str:
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    candidates = []
    for block in code_blocks:
        block = block.strip()
        if block.startswith("[") and block.endswith("]"):
            candidates.append(block)
        else:
            objs = re.findall(r"\{[^{}]*\}", block)
            if objs:
                candidates.append("[" + ",".join(objs) + "]")
    if not candidates:
        objs = re.findall(r"\{[^{}]*\}", s)
        if objs:
            candidates.append("[" + ",".join(objs) + "]")
    for cand in candidates:
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    return "[]"

def parse_features_from_text(generated_text: str):
    raw = extract_json_array_block(generated_text)
    try:
        arr = json.loads(raw)
    except Exception:
        return []
    feats = []
    for x in arr:
        if isinstance(x, dict):
            feat = str(x.get("feature", "")).strip()
            if feat:
                feats.append(feat)
    return feats

# =========================
# Prompts
# =========================
def build_abs_prompt(item_text_blob: str) -> str:
    return f"""
You are a helpful assistant for review compression.

Task:
Given all user reviews about the SAME product/business below, write a HIGH information-density abstract (3–6 short sentences).
- Keep only the key facts and recurring opinions that help identify user concerns.
- Remove repetition, anecdotes, greetings.
- Do NOT output lists, bullets, or JSON.

Reviews:
\"\"\"{item_text_blob}\"\"\"

Output:
""".strip()

def build_item_aspect_prompt(item_abstract: str) -> str:
    return f"""
You are an information extractor.

From the single abstract below, extract a set of distinct review perspectives ("features") FOR THIS ITEM ONLY.
Output ONLY a JSON array, where each element is an object with a single field "feature".

Example:
[
  {{"feature": "food quality"}},
  {{"feature": "service"}},
  {{"feature": "price"}}
]

Abstract:
\"\"\"{item_abstract}\"\"\"

Notes:
- Output ONLY JSON, no extra text.
- Keep each feature concise (1–3 words when possible).
- If nothing is relevant, return [].
""".strip()

# =========================
# 分桶：短/长 item
# =========================
def estimate_item_tokens(tokenizer, texts):
    joined = "\n\n".join(clean_text(t) for t in texts if t and t.strip())
    return count_tokens(tokenizer, joined)

def split_items_by_length(tokenizer, item_to_reviews, long_threshold=LONG_ITEM_TOKENS):
    short_ids, long_ids = [], []
    for iid, texts in item_to_reviews.items():
        n_tok = estimate_item_tokens(tokenizer, texts)
        (short_ids if n_tok <= long_threshold else long_ids).append(iid)
    return short_ids, long_ids

# =========================
# 层次化摘要子组件
# =========================
def chunk_texts_by_token_budget(tokenizer, texts, per_chunk_budget_tokens):
    chunks, cur, cur_tokens = [], [], 0
    for t in texts:
        t = t or ""
        if not t.strip():
            continue
        t_tokens = count_tokens(tokenizer, t)
        if t_tokens > per_chunk_budget_tokens:
            if cur:
                chunks.append(cur); cur, cur_tokens = [], 0
            enc = tokenizer.encode(t, add_special_tokens=False)
            start = 0
            split_cnt = 0
            while start < len(enc) and split_cnt < MAX_CHUNKS_PER_ITEM:
                end = min(start + per_chunk_budget_tokens, len(enc))
                piece = tokenizer.decode(enc[start:end], skip_special_tokens=True)
                chunks.append([piece])
                start = end
                split_cnt += 1
            continue
        if cur_tokens + t_tokens <= per_chunk_budget_tokens:
            cur.append(t); cur_tokens += t_tokens
        else:
            chunks.append(cur); cur, cur_tokens = [t], t_tokens
        if len(chunks) >= MAX_CHUNKS_PER_ITEM:
            if cur: chunks.append(cur)
            break
    if cur:
        chunks.append(cur)
    return chunks

# ---------- 短 item：跨 item 批式直接摘要 ----------
def summarize_short_items_batched(tokenizer, llm, sampling_params_abs, item_to_reviews, item_ids):
    pfx = build_abs_prompt("")
    ids, prompts = [], []
    for iid in item_ids:
        texts = [clean_text(t) for t in item_to_reviews[iid] if t and t.strip()]
        if not texts:
            continue
        blob = "\n\n".join(texts)
        payload = truncate_to_fit(tokenizer, prefix=pfx, payload=blob, budget_tokens=max_model_len)
        ids.append(iid)
        prompts.append(build_abs_prompt(payload))

    abstracts = {}
    if SHOW_PROMPT_PROGRESS:
        tqdm.write(f"P_abs SHORT per-item batched: total={len(prompts)}")
    for s in tqdm(range(0, len(prompts), MAX_PROMPTS_PER_CALL_ABS), desc="SHORT items: abstracts"):
        batch_ids = ids[s:s+MAX_PROMPTS_PER_CALL_ABS]
        batch_prompts = prompts[s:s+MAX_PROMPTS_PER_CALL_ABS]
        outs = llm.generate(batch_prompts, sampling_params_abs)
        for iid, out in zip(batch_ids, outs):
            abstracts[iid] = (out.outputs[0].text if out and out.outputs else "").strip()
    return abstracts

# ---------- 长 item：块级并发 + 层级并发合并（全程跨 item 批） ----------
def build_blocks_for_long_items(tokenizer, item_to_reviews, long_item_ids, per_chunk_budget=CHUNK_TOKEN_BUDGET):
    """返回扁平块列表：[(iid, block_idx, block_text), ...]"""
    flat_blocks = []
    for iid in long_item_ids:
        texts = [clean_text(t) for t in item_to_reviews[iid] if t and t.strip()]
        chunks = chunk_texts_by_token_budget(tokenizer, texts, per_chunk_budget_tokens=per_chunk_budget)
        for bi, block in enumerate(chunks):
            blob = "\n\n".join(block)
            flat_blocks.append((iid, bi, blob))
    return flat_blocks

def summarize_blocks_batched(tokenizer, llm, sampling_params_abs, flat_blocks):
    """块级并发摘要：输入扁平块，输出 {iid: {bi: block_summary}}"""
    pfx = build_abs_prompt("")
    prompts, keys = [], []
    for iid, bi, blob in flat_blocks:
        payload = truncate_to_fit(tokenizer, prefix=pfx, payload=blob, budget_tokens=max_model_len)
        prompts.append(build_abs_prompt(payload))
        keys.append((iid, bi))

    block_summ = defaultdict(dict)
    if SHOW_PROMPT_PROGRESS:
        tqdm.write(f"P_abs LONG block-level batched: total={len(prompts)}")
    for s in tqdm(range(0, len(prompts), MAX_PROMPTS_PER_CALL_ABS), desc="LONG: block summaries"):
        batch_prompts = prompts[s:s+MAX_PROMPTS_PER_CALL_ABS]
        batch_keys    = keys[s:s+MAX_PROMPTS_PER_CALL_ABS]
        outs = llm.generate(batch_prompts, sampling_params_abs)
        for (iid, bi), out in zip(batch_keys, outs):
            txt = (out.outputs[0].text if out and out.outputs else "").strip()
            block_summ[iid][bi] = txt
    return block_summ

def merge_summaries_batched(tokenizer, llm, sampling_params_abs, per_item_segments, merge_budget=MERGE_TOKEN_BUDGET):
    """
    per_item_segments: {iid: [seg1, seg2, ...]}
    并发合并一层，返回新的 {iid: [merged1, merged2, ...]}（可能每个 iid 变成更少段）
    """
    jobs = []  # (iid, gi, merged_blob)
    for iid, segs in per_item_segments.items():
        groups = chunk_texts_by_token_budget(tokenizer, segs, per_chunk_budget_tokens=merge_budget)
        for gi, group in enumerate(groups):
            jobs.append((iid, gi, "\n\n".join(group)))

    pfx = build_abs_prompt("")
    prompts, keys = [], []
    for iid, gi, blob in jobs:
        payload = truncate_to_fit(tokenizer, prefix=pfx, payload=blob, budget_tokens=max_model_len)
        prompts.append(build_abs_prompt(payload))
        keys.append((iid, gi))

    per_item_next = defaultdict(list)
    if SHOW_PROMPT_PROGRESS:
        tqdm.write(f"P_abs LONG merge-level batched: total={len(prompts)}")
    for s in tqdm(range(0, len(prompts), MAX_PROMPTS_PER_CALL_MERGE), desc="LONG: merge level"):
        batch_prompts = prompts[s:s+MAX_PROMPTS_PER_CALL_MERGE]
        batch_keys    = keys[s:s+MAX_PROMPTS_PER_CALL_MERGE]
        outs = llm.generate(batch_prompts, sampling_params_abs)
        for (iid, gi), out in zip(batch_keys, outs):
            txt = (out.outputs[0].text if out and out.outputs else "").strip()
            if txt:
                per_item_next[iid].append(txt)
    return per_item_next

def summarize_long_items_hier_batched(tokenizer, llm, sampling_params_abs, item_to_reviews, long_item_ids):
    # 1) 扁平化块，块级并发摘要
    flat_blocks = build_blocks_for_long_items(tokenizer, item_to_reviews, long_item_ids)
    block_summ  = summarize_blocks_batched(tokenizer, llm, sampling_params_abs, flat_blocks)  # {iid: {bi: text}}
    # 2) 逐层并发合并，直到每个 iid 只剩 1 段或达到 MAX_LEVELS
    per_item_segments = {iid: [txt for _, txt in sorted(d.items())] for iid, d in block_summ.items()}
    level = 1
    while level <= MAX_LEVELS:
        if all(len(v) <= 1 for v in per_item_segments.values()):
            break
        per_item_segments = merge_summaries_batched(
            tokenizer, llm, sampling_params_abs, per_item_segments, merge_budget=MERGE_TOKEN_BUDGET
        )
        level += 1
    # 3) 收尾：取每个 iid 的第一段作为最终摘要
    abstracts = {iid: (segs[0] if segs else "") for iid, segs in per_item_segments.items()}
    return abstracts

# =========================
# 逐 item 抽方面（批量并发）
# =========================
def build_item_aspect_prompt_safe(tokenizer, abstract_text: str) -> str:
    # 保险截断（几乎用不到，但留着更稳）
    pfx = build_item_aspect_prompt("")
    payload = truncate_to_fit(tokenizer, prefix=pfx, payload=abstract_text, budget_tokens=max_model_len)
    return build_item_aspect_prompt(payload)

def extract_item_aspects_batched(tokenizer, llm, sampling_params_aspect, item_abstracts):
    """
    输入：{item_id: abstract_text}
    输出：(item_aspects, aspect_item_dfreq)
      - item_aspects: {item_id: [aspect1, aspect2, ...]}
      - aspect_item_dfreq: Counter，记录每个方面出现在多少个不同 item 中
    """
    ids, prompts = [], []
    for iid, abst in item_abstracts.items():
        if not abst or not abst.strip():
            continue
        ids.append(iid)
        prompts.append(build_item_aspect_prompt_safe(tokenizer, abst))

    if SHOW_PROMPT_PROGRESS:
        tqdm.write(f"P_aspect per-item: total items with non-empty abstracts = {len(prompts)}")

    item_aspects = {}
    for start in tqdm(range(0, len(prompts), MAX_PROMPTS_PER_CALL_ASPECT),
                      desc="P_aspect per-item (batched)"):
        batch_ids = ids[start:start+MAX_PROMPTS_PER_CALL_ASPECT]
        batch_prompts = prompts[start:start+MAX_PROMPTS_PER_CALL_ASPECT]
        outs = llm.generate(batch_prompts, sampling_params_aspect)
        for iid, out in zip(batch_ids, outs):
            gen = out.outputs[0].text.strip() if out and out.outputs else ""
            feats = [f.lower() for f in parse_features_from_text(gen)]
            feats = [re.sub(r"\s+", " ", f).strip(" .,:;") for f in feats if f.strip()]
            uniq, seen = [], set()
            for f in feats:
                if f not in seen:
                    seen.add(f)
                    uniq.append(f)
            item_aspects[iid] = uniq

    dfreq = Counter()
    for iid, feats in item_aspects.items():
        for f in set(feats):
            dfreq[f] += 1
    return item_aspects, dfreq

# =========================
# 主流程
# =========================
if __name__ == "__main__":
    random.seed(random_seed)
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    model_path = '/disk3/zheng'

    tokenizer, llm = init_model(model_name, model_path, batch_size=64, max_model_len=max_model_len)

    sampling_params_abs = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=abs_max_tokens)
    sampling_params_aspect = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=aspect_max_tokens)

    # ===== 读取数据 =====
    data_path = '/home/zheng/reviewgpt/aspect_extraction/filtered_yelp_restaurant_reviews.jsonl'
    reviews = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                rating = r.get("stars")
                text   = r.get("text", "")
                asin   = r.get("business_id")  # item id
                uid    = r.get("user_id")
                if asin and text and str(text).strip():
                    reviews.append({"rating": rating, "text": text, "asin": asin, "user_id": uid})
            except json.JSONDecodeError:
                continue

    logging.info(f"总评论条数: {len(reviews)}")

    # ===== 按 item 聚合 =====
    item_to_reviews_all = defaultdict(list)
    for r in reviews:
        item_to_reviews_all[r["asin"]].append(r["text"])

    all_items = list(item_to_reviews_all.keys())
    logging.info(f"唯一 item 数: {len(all_items)}")

    # （可选）对 item 再次总体下采样
    if sample_fraction < 1.0:
        keep_n = max(1, int(len(all_items) * sample_fraction))
        all_items = random.sample(all_items, keep_n)
        item_to_reviews_all = {iid: item_to_reviews_all[iid] for iid in all_items}
        logging.info(f"下采样后 item 数: {len(all_items)}")

    # ===== 分桶：短 / 长 =====
    short_ids, long_ids = split_items_by_length(tokenizer, item_to_reviews_all, long_threshold=LONG_ITEM_TOKENS)
    logging.info(f"短 item: {len(short_ids)}，长 item: {len(long_ids)}（阈值 LONG_ITEM_TOKENS={LONG_ITEM_TOKENS}）")

    # ===== 摘要：短 item（跨 item 批）=====
    abs_short = summarize_short_items_batched(
        tokenizer, llm, sampling_params_abs, item_to_reviews_all, short_ids
    )

    # ===== 摘要：长 item（块级并发 + 层级并发合并，且跨 item 批）=====
    abs_long = summarize_long_items_hier_batched(
        tokenizer, llm, sampling_params_abs, item_to_reviews_all, long_ids
    )

    # 合并为“每个 item 恰好 1 条摘要”
    item_abstracts = {**abs_short, **abs_long}
    logging.info(f"得到摘要的 item 数: {len(item_abstracts)}")

    # ===== 抽方面：逐 item 批量并发 =====
    item_aspects, aspect_item_dfreq = extract_item_aspects_batched(
        tokenizer, llm, sampling_params_aspect, item_abstracts
    )

    # ===== 共识：基于“出现在多少个 item 中” =====
    consensus_kept = sorted([a for a, d in aspect_item_dfreq.items() if d >= consensus_tau_items])
    logging.info(f"满足共识阈值(≥{consensus_tau_items} 个 item)的方面数: {len(consensus_kept)}")

    # ===== 输出与保存 =====
    out_dir = '/home/zheng/reviewgpt/aspect_extraction'
    os.makedirs(out_dir, exist_ok=True)

    # 每个 item 的方面
    path_item_aspects = os.path.join(out_dir, "item_aspects.json")
    with open(path_item_aspects, "w", encoding="utf-8") as f:
        json.dump(item_aspects, f, ensure_ascii=False, indent=2)

    # 方面的 item 文档频（出现于多少个 item）
    # 注意：Counter 不能直接 JSON 序列化，这里转成普通 dict
    path_dfreq = os.path.join(out_dir, "aspect_dfreq.json")
    with open(path_dfreq, "w", encoding="utf-8") as f:
        json.dump({
            "consensus_tau_items": consensus_tau_items,
            "aspect_item_dfreq": dict(aspect_item_dfreq),
            "consensus_vocab": consensus_kept
        }, f, ensure_ascii=False, indent=2)

    print("=== 全局共识方面（按字典序） ===")
    for a in consensus_kept:
        print(f"- {a}")

    print(f"每个 item 的方面已保存：{path_item_aspects}")
    print(f"全局方面文档频与共识词表已保存：{path_dfreq}")
