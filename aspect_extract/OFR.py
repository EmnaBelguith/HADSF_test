#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from math import ceil
from multiprocessing import Process, Queue
from typing import List, Tuple

import torch
from tqdm import tqdm

# ------------------- 1. 块级处理函数（独立进程） -------------------
def process_chunk(jsonl_path: str, start: int, end: int, worker_id: int, result_queue: Queue,
                  delta: int = 2, max_len: int = 512):
    """
    按公式计算 OFR：
      OFR = (1/|S|) * sum_{(u,i,τ,s)∈S} ( 1/|s| * sum_{(a,o,s)∈s} SemSim(o,r) )
    其中 SemSim(o,r)= max_{r_{s:e}∈S_{L(o),Δ}(r)} < ψ(o), ψ(r_{s:e}) >/ (||ψ(o)||·||ψ(r_{s:e})||)

    参数:
      delta: Δ，长度窗口半宽
      max_len: 编码时的最大 token 长度上限（截断）
    """
    # 1) 本进程设备选择
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = worker_id % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"[Worker {worker_id}] 使用设备: {device}")

    # 2) 在进程内部加载依赖
    import spacy
    from transformers import AutoTokenizer, AutoModel

    # 用小型英文分词器做**词级**滑窗
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer"])
    nlp.add_pipe("sentencizer")

    # 用底座 Transformer，按“均值池化 token”实现 ψ(·)
    # 说明：使用 sentence-transformers/all-MiniLM-L6-v2 的底座做 last_hidden_state 平均，贴合公式定义
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name).to(device)
    backbone.eval()

    @torch.no_grad()
    def mean_pool_embeddings(texts: List[str]) -> torch.Tensor:
        """
        按公式 ψ(x_{1:T})= (1/T) sum_t φ(x_t)，这里用 last_hidden_state 与 attention_mask 做**掩码均值**。
        返回 L2 归一化后的向量，便于直接点积=余弦。
        """
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)
        out = backbone(**enc).last_hidden_state  # [B, L, H]
        mask = enc.attention_mask.unsqueeze(-1)  # [B, L, 1]
        summed = (out * mask).sum(dim=1)         # [B, H]
        counts = mask.sum(dim=1).clamp(min=1)    # [B, 1]
        mean = summed / counts                   # [B, H]
        # L2 归一化
        mean = torch.nn.functional.normalize(mean, p=2, dim=-1)
        return mean  # [B, H], 单位向量

    def make_contiguous_spans(text: str, L: int, Delta: int) -> List[str]:
        """
        S_{L,Δ}(r): 基于**词级**的连续窗口，长度落在 [L-Δ, L+Δ]。
        用 spaCy 的 token 序列做滑窗，回到文本用空格 join（与 opinion 的 token 粒度一致性足够用于度量）。
        """
        doc = nlp(text)
        words = [t.text for t in doc]  # 保留所有 token（包含标点），与定义更一致
        n = len(words)
        if n == 0:
            return []

        min_len = max(1, L - Delta)
        max_len_win = max(min_len, L + Delta)

        spans: List[str] = []
        for win in range(min_len, max_len_win + 1):
            if win > n:
                break
            for i in range(0, n - win + 1):
                span = " ".join(words[i:i + win])
                spans.append(span)
        return spans

    def semsim(opinion: str, review: str) -> float:
        """
        严格按定义：
          - 若 opinion 在 review 中逐字出现（不区分大小写），直接返回 1.0
          - 否则枚举 S_{L(o),Δ}(r) 的所有连续窗口，取最大余弦相似度
        """
        if not opinion or not review:
            return 0.0

        # 逐字包含判定
        if opinion.lower() in review.lower():
            return 1.0

        L_o = max(1, len(opinion.split()))
        cands = make_contiguous_spans(review, L_o, delta)
        if not cands:
            return 0.0

        # 计算 ψ(o) 与 ψ(spans) 并取最大归一化点积
        vec_o = mean_pool_embeddings([opinion])           # [1, H], 已归一化
        vec_s = mean_pool_embeddings(cands)               # [K, H], 已归一化
        sims = (vec_s @ vec_o.T).squeeze(-1)              # [K]
        max_sim = float(torch.max(sims).item())
        return max_sim

    # 3) 逐行处理本区间并显示进度条，按“先内层再外层平均”聚合
    review_avg_sum = 0.0    # 本进程所有 review 的“(1/|s|)∑ SemSim”之和
    review_count   = 0      # 本进程中 |s|>0 的 review 数
    chunk_size = end - start

    with open(jsonl_path, "r", encoding="utf-8") as f:
        # 跳过前 start 行
        for _ in range(start):
            if not f.readline():
                break

        pbar = tqdm(total=chunk_size,
                    desc=f"Worker {worker_id}",
                    position=worker_id,
                    leave=True,
                    unit="line")
        for _ in range(chunk_size):
            line = f.readline()
            if not line:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                pbar.update(1)
                continue

            review = data.get("text", "")
            raw_triples = data.get("sentence", [])

            # —— 去重 (a,o,s) 三元组 —— 
            seen = set()
            triples = []
            for triple in raw_triples:
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                key = tuple(triple)
                if key not in seen:
                    seen.add(key)
                    triples.append(triple)

            # 对当前 review：内层平均
            sim_sum = 0.0
            tri_cnt = 0
            for (aspect, opinion, sentiment) in triples:
                if not isinstance(opinion, str):
                    continue
                sim_sum += semsim(opinion, review)
                tri_cnt += 1

            if tri_cnt > 0:
                review_avg_sum += (sim_sum / tri_cnt)
                review_count   += 1

            pbar.update(1)

        pbar.close()

    # 4) 汇总到主进程
    result_queue.put((review_avg_sum, review_count))


# ------------------- 2. 主进程：划分任务 & 启动子进程 -------------------
def compute_ofr_parallel(jsonl_path: str, n_procs: int = None, delta: int = 2):
    print(f"现在在处理 {jsonl_path} 数据……")

    # 统计总行数
    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # 决定进程数（优先不超过 GPU 数；无 GPU 时默认 8）
    visible_gpus = torch.cuda.device_count()
    n_procs = n_procs or (visible_gpus if visible_gpus > 0 else 8)
    n_procs = max(1, min(n_procs, total_lines))  # 不超过行数

    chunk_size = ceil(total_lines / n_procs)
    result_queue = Queue()
    processes = []

    for i in range(n_procs):
        start = i * chunk_size
        end   = min((i + 1) * chunk_size, total_lines)
        if start >= end:
            break
        p = Process(
            target=process_chunk,
            args=(jsonl_path, start, end, i, result_queue),
            kwargs=dict(delta=delta)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 汇总
    total_review_avg_sum = 0.0
    total_review_count   = 0
    while not result_queue.empty():
        rev_sum, rev_cnt = result_queue.get()
        total_review_avg_sum += rev_sum
        total_review_count   += rev_cnt

    if total_review_count == 0:
        print("无包含三元组的 review，无法计算 OFR")
    else:
        ofr = total_review_avg_sum / total_review_count
        print(f"处理 {jsonl_path} 数据……")
        print(f"包含三元组的 review 数 : {total_review_count}")
        print(f"OFR （先内层再外层平均）: {ofr:.4f}")


# ------------------- 3. 调用示例 --------------------
if __name__ == "__main__":
    jsonl_file = (
        "/home/infres/belguith/HADSF_test/output/reviews_with_aspects.jsonl"
    )
    # n_procs：建议与可见 GPU 数一致；Δ 可按需要调整
    compute_ofr_parallel(jsonl_file, n_procs=8, delta=2)
