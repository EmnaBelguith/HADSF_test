# -*- coding: utf-8 -*-
"""
validate Bert-Whitening: https://kexue.fm/archives/8069
"""
import sys
import os

# 获取当前脚本所在的目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import pandas as pd# -*- coding: utf-8 -*-
"""
validate Bert-Whitening: https://kexue.fm/archives/8069
"""
import sys
import os

# 获取当前脚本所在的目录的父目录
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import pandas as pd
import torch.nn.functional as func
import argparse
import copy
from abc import ABC
from collections import Counter, defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig, ElectraTokenizerFast
from load_data import load_sentiment_data, get_dir_and_base_name, load_aspect_data  # 确保这些模块已正确定义
import numpy as np
from util import get_logger, get_args_str, get_tensorboard_writer  # 确保这些模块已正确定义
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from nltk.tokenize import sent_tokenize
import nltk
from infomap import Infomap
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle
import json
from sklearn.preprocessing import LabelEncoder  # 新增

# 初始化日志
logger = get_logger('BERT-Whitening', None)

NO_ASPECT = "no_aspect"

parser = argparse.ArgumentParser()
parser.add_argument('--vec_dim', type=int, default=128)

parser.add_argument('--device', type=str, default='cuda:7')
parser.add_argument('--dataset_name', type=str, default='Instant_Video_5',
                    help='dataset name')
parser.add_argument('--dataset_path', type=str, default='/home/d1/shuaijie/data/Instant_Video_5/Instant_Video_5.json',
                    help='raw dataset file path')

parser.add_argument('--review_max_length', type=int, default=128)

args = parser.parse_args()
# filtered_Industrial_and_Scientific_output_aspect_qwen7b
# 设置参数
args.dataset_name = 'Musical_Instruments_aspect_20as_test'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output_aspect_20as/filtered_Musical_Instruments_0506_20as_test.jsonl' # 修改为您的 JSONL 文件路径
args.dataset_name = 'Musical_Instruments_aspect_20as'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output_aspect_20as/filtered_Musical_Instruments_0506_20as.jsonl' # 修改为您的 JSONL 文件路径
args.dataset_name = 'Musical_Instruments_aspect_5as'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output_aspect_5as/filtered_Musical_Instruments_0506_5as.jsonl' # 修改为您的 JSONL 文件路径
# args.dataset_name = 'Industrial_and_Scientific_5as'
# args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Industrial_and_Scientific_5as/filtered_Industrial_and_Scientific_5as.jsonl' # 修改为您的 JSONL 文件路径
args.dataset_name = 'yelp_reviews_5as'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_yelp_reviews_output_aspect_5as/filtered_yelp_restaurant_reviews_5as.jsonl' # 修改为您的 JSONL 文件路径
args.dataset_name = 'Musical_Instruments_0506_10as'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_10as/filtered_Musical_Instruments_0506_10as.jsonl' # 修改为您的 JSONL 文件路径
args.dataset_name = 'Musical_Instruments_aspect'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output_aspect/filtered_Musical_Instruments_output.jsonl' # 修改为您的 JSONL 文件路径
args.dataset_name = 'yelp_reviews_aspect_part2'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_yelp_reviews_aspect_10-50/filtered_yelp_reviews_aspect_10-50.jsonl' # 修改为您的 JSONL 文件路径
args.dataset_name = 'Musical_Instruments_aspect_part1'
args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_aspect_0-10/filtered_Musical_Instruments_aspect_0-10.jsonl' # 修改为您的 JSONL 文件路径
# args.dataset_name = 'Industrial_and_Scientific_aspect_part3'
# args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Industrial_and_Scientific_aspect_50-100/filtered_Industrial_and_Scientific_aspect_50-100.jsonl' # 修改为您的 JSONL 文件路径

# yelp_reviews_qwen_32b_aspect
args.vec_dim = 128

args.model_short_name = 'BERT-Whitening'
args.pretrained_weight_shortcut = '/home/zheng/reviewgpt/aspect_extraction/RGCL/models/bert-base-uncased'  # 修改为您的 BERT 模型路径
model_name = os.path.basename(args.pretrained_weight_shortcut)

args.feat_save_path = os.path.join(
    '.', 'checkpoint',
    args.dataset_name,
    args.model_short_name,
    f'{model_name}_sentence_vectors_dim_{args.vec_dim}.pkl'
)
save_dir = os.path.dirname(args.feat_save_path)
os.makedirs(save_dir, exist_ok=True)

# 初始化 BERT tokenizer
bert_tokenizer = ElectraTokenizerFast.from_pretrained(args.pretrained_weight_shortcut,
                                               model_max_length=args.review_max_length)


class ReviewDataset(Dataset):

    def __init__(self, user, item, rating, review_text, tokenizer):
        self.user = np.array(user).astype(np.int64)
        self.item = np.array(item).astype(np.int64)
        self.r = np.array(rating).astype(np.float32)
        self.tokenizer = tokenizer
        self.docs = review_text
        self.aspects = []  # 新增，用于存储每个句子的 aspect

        # 确保所有列表长度一致
        assert len(self.user) == len(self.item) == len(self.r) == len(self.docs), \
            f"Lengths do not match: user={len(self.user)}, item={len(self.item)}, rating={len(self.r)}, docs={len(self.docs)}"

        self.__pre_tokenize()

    # 将评论文本分割为句子，并根据长度限制句子
    def __pre_tokenize(self):
        self.docs = [self.tokenizer.tokenize(x) for x in tqdm(self.docs, desc='Pre-tokenizing reviews')]
        review_length = self.top_review_length(self.docs)
        self.docs = [x[:review_length] for x in tqdm(self.docs, desc='Truncating sentences')]

    # 返回指定索引的数据
    def __getitem__(self, idx):
        return self.user[idx], self.item[idx], self.r[idx], self.docs[idx]

    def __len__(self):
        return len(self.docs)

    @staticmethod
    def top_review_length(docs: list, top=0.8):
        sentence_length = [len(x) for x in docs]
        sentence_length.sort()
        length = sentence_length[int(len(sentence_length) * top)]
        length = 256 if length > 256 else length
        return length


def collate_fn(data):
    # 解包数据
    u, i, r, tokens = zip(*data)
    if isinstance(tokens, tuple):
        tokens = list(tokens)

    # 使用 BERT tokenizer 进行编码
    encoding = bert_tokenizer(tokens, return_tensors='pt', padding=True,
                              truncation=True, is_pretokenized=True)

    return torch.Tensor(u), torch.Tensor(i), torch.Tensor(r), \
        encoding['input_ids'], encoding['attention_mask']


# 计算 kernel 和 bias
def compute_kernel_bias(vecs, vec_dim):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :vec_dim], -mu


# 应用变换并标准化
def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化"""
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def load_jsonl(file_path):
    """加载 JSONL 文件到 Pandas DataFrame，并重命名字段以适应后续处理"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Loading JSONL data'):
            json_line = json.loads(line)
            # 重命名字段
            json_line['user_id'] = json_line.pop('reviewerID')
            json_line['review_text'] = json_line.pop('reviewText')
            json_line['item_id'] = json_line.pop('asin')
            json_line['rating'] = json_line.pop('overall')
            json_line['timestamp'] = json_line.pop('unixReviewTime')
            # 如果需要，可以添加其他字段的处理
            data.append(json_line)
    df = pd.DataFrame(data)
    return df


@torch.no_grad()
def save_sentence_feat(params):
    if os.path.exists(params.feat_save_path):
        print(f"向量文件已存在，正在从 {params.feat_save_path} 加载向量和预处理数据...")
        vecs_dict = torch.load(params.feat_save_path)
        if not isinstance(vecs_dict, dict):
            raise ValueError(f"加载的 vecs 不是字典，实际类型为 {type(vecs_dict)}")
        preprocessed_data_path = params.feat_save_path.replace('.pkl', '_preprocessed.pkl')
        if not os.path.exists(preprocessed_data_path):
            raise FileNotFoundError(f"预处理数据文件未找到: {preprocessed_data_path}")
        with open(preprocessed_data_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        sentences_df = preprocessed_data['sentences_df']
        df = preprocessed_data['df']
        print("预处理数据已加载。")
    else:
        # 加载数据
        print("Loading and processing JSONL data...")
        train_data, valid_data, test_data, word2id, embeddings, _ = \
            load_aspect_data(args.dataset_path)

        df = pd.concat([train_data, valid_data, test_data])


        # 分配唯一的 review_id
        df = df.reset_index(drop=True)
        df['review_id'] = df.index

        # 分割评论文本为句子
        print("Splitting reviews into sentences...")
#        nltk.download('punkt')  # 确保 punkt 已下载
        df['sentences'] = df['review_text'].apply(sent_tokenize)
        sentences = []
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing reviews'):
            user_id = row['user_id']
            item_id = row['item_id']
            review_id = row['review_id']
            aspects = row['aspect']  # aspect 格式: [["aspect_name","opinion_phrase",...], ...]


            # 构建 aspect 到 opinion_phrase 的映射
            aspect_opinions = {aspect[0]: aspect[1] for aspect in aspects} if aspects else {}

            for sentence in row['sentences']:
                assigned_aspects = []
                for aspect, opinion in aspect_opinions.items():
                    # 采用不区分大小写的匹配
                    if opinion.lower() in sentence.lower():
                        assigned_aspects.append(aspect)
                if assigned_aspects:
                    for aspect in assigned_aspects:
                        sentences.append({
                            'sentence_id': len(sentences),
                            'review_id': review_id,
                            'user_id': user_id,
                            'item_id': item_id,
                            'sentence': sentence,
                            'aspect': aspect  # 分配的主题
                        })
                else:
                    # 过滤没有对应 aspect 的句子
                    sentences.append({
                        'sentence_id': len(sentences),
                        'review_id': review_id,
                        'user_id': user_id,
                        'item_id': item_id,
                        'sentence': sentence,
                        'aspect': NO_ASPECT  # 分配到 "no_aspect"
                    })

        sentences_df = pd.DataFrame(sentences)

        print(f'sentences_df length: {len(sentences_df)}')
        print(f'user_id length: {len(sentences_df["user_id"])}')
        print(f'item_id length: {len(sentences_df["item_id"])}')
        print(f'review_id length: {len(sentences_df["review_id"])}')
        print(f'sentence length: {len(sentences_df["sentence"])}')
        print(f'aspect length: {len(sentences_df["aspect"])}')

        # 创建 ReviewDataset
        print("Creating ReviewDataset...")
        review_dataset = ReviewDataset(
            sentences_df['user_id'].tolist(),
            sentences_df['item_id'].tolist(),
            sentences_df['review_id'].tolist(),
            sentences_df['sentence'].tolist(),
            bert_tokenizer
        )
        review_dataset.aspects = sentences_df['aspect'].tolist()

        print(f'Length of dataset: {len(review_dataset)}')
        print(f'Length of user: {len(review_dataset.user)}')
        print(f'Length of item: {len(review_dataset.item)}')
        print(f'Length of review_id: {len(review_dataset.r)}')
        print(f'Length of docs: {len(review_dataset.docs)}')
        print(f'Length of aspects: {len(review_dataset.aspects)}')

        # 创建 DataLoader
        print("Creating DataLoader...")
        data_loader = DataLoader(review_dataset, batch_size=250, collate_fn=collate_fn, drop_last=False)

        # 加载 BERT 模型
        print("Loading BERT model...")
        config = BertConfig.from_pretrained(params.pretrained_weight_shortcut)
        config.output_hidden_states = True
        config.return_dict = True
        bert = BertModel.from_pretrained(params.pretrained_weight_shortcut, config=config).to(params.device)
        bert.eval()

        # 编码句子并生成向量
        print("Encoding sentences with BERT...")
        sentence_vecs = []
        sentence_aspects = []
        for u, i, r, input_ids, mask in tqdm(data_loader, desc='Encoding batches'):
            input_ids = input_ids.to(params.device)
            mask = mask.to(params.device)

            with torch.no_grad():
                outputs = bert(input_ids=input_ids, attention_mask=mask)
                output1 = outputs.hidden_states[-2]
                output2 = outputs.hidden_states[-1]
                last2 = (output1 + output2) / 2
                last2 = torch.sum(mask.unsqueeze(-1) * last2, dim=1) / mask.sum(dim=1, keepdim=True)
                sentence_vecs.append(last2.cpu())

            # 存储对应的 aspect 信息
            sentence_aspects_batch = review_dataset.aspects[:input_ids.size(0)]
            sentence_aspects.extend(sentence_aspects_batch)
            review_dataset.aspects = review_dataset.aspects[input_ids.size(0):]

        sentence_vecs = torch.cat(sentence_vecs, dim=0)  # 变为 (num_sentences, vec_dim)

        ##########
        print("Creating Review-Level ReviewDataset by merging sentences...")
        # 按 review_id 合并句子
        df_reviews = df[['review_id', 'user_id', 'item_id', 'sentences']].reset_index(drop=True)
        merged_reviews = df_reviews['sentences'].apply(lambda sents: ' '.join(sents)).tolist()

        print("Creating ReviewDataset for merged reviews...")
        review_level_dataset = ReviewDataset(
            df_reviews['user_id'].tolist(),
            df_reviews['item_id'].tolist(),
            df_reviews['rating'].tolist() if 'rating' in df_reviews.columns else [1.0]*len(df_reviews),
            merged_reviews,
            bert_tokenizer
        )

        print(f'Length of review-level dataset: {len(review_level_dataset)}')
        print(f'Length of user: {len(review_level_dataset.user)}')
        print(f'Length of item: {len(review_level_dataset.item)}')
        print(f'Length of rating: {len(review_level_dataset.r)}')
        print(f'Length of docs: {len(review_level_dataset.docs)}')

        # 创建 DataLoader for reviews
        print("Creating DataLoader for reviews...")
        review_data_loader = DataLoader(review_level_dataset, batch_size=2500, collate_fn=collate_fn, drop_last=True)

        # 编码评论并生成向量
        print("Encoding reviews with BERT...")
        review_vecs = []
        for u, i, r, input_ids, mask in tqdm(review_data_loader, desc='Encoding review batches'):
            # 将数据移至 GPU
            input_ids = input_ids.to(params.device)
            mask = mask.to(params.device)  # bs * seq_len

            # 获取 BERT 输出
            with torch.no_grad():
                outputs = bert(input_ids=input_ids, attention_mask=mask)
                output1 = outputs.hidden_states[-2]  # bs * seq_len * 768
                output2 = outputs.hidden_states[-1]  # bs * seq_len * 768
                last2 = (output1 + output2) / 2
                # 对所有 token 的隐藏状态进行加权求和
                last2 = torch.sum(mask.unsqueeze(-1) * last2, dim=1) \
                    / mask.sum(dim=1, keepdim=True)
                review_vecs.append(last2.cpu())

        review_vecs = torch.cat(review_vecs, dim=0)  # 变为 (num_reviews, vec_dim)

        # 计算 kernel 和 bias
        print("Computing kernel and bias for whitening...")
       # 将句子和评论嵌入一起用于计算白化参数
        all_vecs = torch.cat([sentence_vecs, review_vecs], dim=0).numpy()
        kernel, bias = compute_kernel_bias(all_vecs, params.vec_dim)


        # 应用 whitening 并标准化
        print("Applying whitening and normalization to sentence embeddings...")
        sentence_vecs_whitened = transform_and_normalize(sentence_vecs.numpy(), kernel, bias)
        sentence_vecs_whitened = torch.from_numpy(sentence_vecs_whitened)

        print("Applying whitening and normalization to review embeddings...")
        review_vecs_whitened = transform_and_normalize(review_vecs.numpy(), kernel, bias)
        review_vecs_whitened = torch.from_numpy(review_vecs_whitened)

        # 保存句子嵌入
        print("Saving sentence embeddings to .npy file...")
        sentence_embedding_path = params.feat_save_path.replace('.pkl', '_whitening_sentence_embedding_128.npy')
        np.save(sentence_embedding_path, sentence_vecs_whitened.numpy())
        print(f'Saved sentence embeddings to {sentence_embedding_path}')

        # 保存评论嵌入
        print("Saving review embeddings to .npy file...")
        review_embedding_path = params.feat_save_path.replace('.pkl', '_whitening_review_embedding_128.npy')
        np.save(review_embedding_path, review_vecs_whitened.numpy())
        print(f'Saved review embeddings to {review_embedding_path}')

        # 保存预处理数据和向量
        print("Saving preprocessed data and BERT-Whitening features...")
        # 使用整数类型的 sentence_id 作为键
        sentence_ids = sentences_df['sentence_id'].tolist()
        vecs_dicts = {sid: vec for sid, vec in zip(sentence_ids, sentence_vecs_whitened)}
        torch.save(vecs_dicts, params.feat_save_path)
        preprocessed_data = {
            'sentences_df': sentences_df,
            'df': df,
        }
        preprocessed_data_path = params.feat_save_path.replace('.pkl', '_preprocessed.pkl')
        with open(preprocessed_data_path, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        print(f'Saved embeddings to {params.feat_save_path}')
        print(f'Saved preprocessed data to {preprocessed_data_path}')


    # 构建 topic_and_sentence.pkl
    # 构建 topic_and_sentence.pkl
    print("Building topic and sentence mappings...")
    aspect_counts = sentences_df['aspect'].value_counts()
    print(f"Aspect counts:\n{aspect_counts}")

    filtered_aspects = aspect_counts[aspect_counts >= 1].index.tolist()
    print(f"Filtered aspects (count >= 1) ({len(filtered_aspects)}): {filtered_aspects}")

    filtered_aspects = [aspect for aspect in filtered_aspects if aspect != NO_ASPECT]
    print(f"Filtered aspects after excluding NO_ASPECT ({len(filtered_aspects)}): {filtered_aspects}")

    unique_aspects = sorted(filtered_aspects)
    print(f"Unique aspects ({len(unique_aspects)}): {unique_aspects}")

    aspect_to_topic_id = {aspect: idx for idx, aspect in enumerate(unique_aspects, start=1)}  # 从1开始
    print(f"Aspect to Topic ID mapping: {aspect_to_topic_id}")    

        # 3. 构建 sid_to_topic 和 topic_to_sid
    sid_to_topic = {}
    topic_to_sid = defaultdict(list)
    filtered_sentences_df = sentences_df[sentences_df['aspect'].isin(unique_aspects)]

    for sid, aspect in zip(filtered_sentences_df['sentence_id'], filtered_sentences_df['aspect']):
 #       if aspect != NO_ASPECT:
 #           topic_id = aspect_to_topic_id[aspect]
 #           sid_to_topic[sid] = topic_id
 #           topic_to_sid[topic_id].append(sid)
        topic_id = aspect_to_topic_id[aspect]
        sid_to_topic[sid] = topic_id
        topic_to_sid[topic_id].append(sid)

        # 4. 构建最终的映射结构
    topic_and_sentence = {
        'sid_to_topic': sid_to_topic,
        'topic_to_sid': dict(topic_to_sid)
    }

    topic_and_sentence_pkl_path = os.path.join(save_dir, 'topic_and_sentence.pkl')
    save_pickle(topic_and_sentence, topic_and_sentence_pkl_path)

    topic_and_sentence_pkl_path = os.path.join(save_dir, 'topic_and_sentence.pkl')
    save_pickle(topic_and_sentence, topic_and_sentence_pkl_path)

    # 构建 ui_to_review_id.pkl
    print("Building UI to Review ID mappings...")
    ui_to_review_id = build_ui_to_review_id(df)
    ui_to_review_id_pkl_path = os.path.join(save_dir, 'ui_to_review_id.pkl')
    save_pickle(ui_to_review_id, ui_to_review_id_pkl_path)

    # 构建 ui_to_sentence_id.pkl
    print("Building UI to Sentence ID mappings...")
    ui_to_sentence_id = build_ui_to_sentence_id(df, sentences_df)
    ui_to_sentence_id_pkl_path = os.path.join(save_dir, 'ui_to_sentence_id.pkl')
    save_pickle(ui_to_sentence_id, ui_to_sentence_id_pkl_path)

    print("All mappings have been successfully created and saved.")


def compute_sentence_similarity(vecs, top_k=10, threshold=0.7, device='cuda', batch_size=512):
    """
    计算句子间的余弦相似度，并构建相似度图，支持 GPU 加速，使用批处理减少内存使用。
    
    :param vecs: dict，键为 sentence_id，值为 torch.Tensor
    :param top_k: 每个句子保留的最近邻数量
    :param threshold: 相似度阈值，低于该值的边将被移除
    :param device: 使用的设备，默认为 'cuda' 表示 GPU
    :param batch_size: 批处理大小
    :return: NetworkX 图
    """
    print("Computing cosine similarity on device:", device)

    # 将所有向量移动到 GPU 上并转换为 float16
    vecs_list = [vec.to(device).half() for vec in vecs.values()]
    vecs_tensor = torch.stack(vecs_list)  # Shape: (num_sentences, vec_dim)
    
    num_sentences = vecs_tensor.size(0)
    G = nx.Graph()
    G.add_nodes_from(range(num_sentences))

    print("Building similarity graph with batching...")

    all_edges = []
    with torch.no_grad():
        for i in tqdm(range(0, num_sentences, batch_size), desc="Building graph"):
            # 获取当前批次的句子向量
            batch_vecs = vecs_tensor[i:i+batch_size]  # Shape: (batch_size, vec_dim)
            
            # 计算余弦相似度矩阵
            # Normalize batch_vecs and vecs_tensor
            batch_norm = torch.norm(batch_vecs, p=2, dim=1, keepdim=True)  # (batch_size, 1)
            all_norm = torch.norm(vecs_tensor, p=2, dim=1, keepdim=True)  # (num_sentences, 1)
            # To prevent division by zero
            batch_norm[batch_norm == 0] = 1e-8
            all_norm[all_norm == 0] = 1e-8
            sim_matrix = torch.mm(batch_vecs, vecs_tensor.T) / (batch_norm * all_norm.T)  # (batch_size, num_sentences)
            
            # Exclude self-similarity by setting diagonal to -1
            end_idx = min(i + batch_size, num_sentences)
            self_indices = torch.arange(i, end_idx).to(device)
            sim_matrix[:, self_indices] = -1.0
            
            # Find top_k similar sentences per sentence in the batch
            topk_scores, topk_indices = torch.topk(sim_matrix, top_k, dim=1, largest=True, sorted=False)  # Both are (batch_size, top_k)
            
            # Apply threshold
            mask = topk_scores >= threshold  # (batch_size, top_k)
            
            # Expand row indices
            batch_size_actual = batch_vecs.size(0)
            row_indices = torch.arange(i, i + batch_size_actual, device=device).unsqueeze(1).expand_as(topk_indices)  # (batch_size, top_k)
            
            # Apply mask
            row_indices = row_indices[mask]
            topk_indices = topk_indices[mask]
            topk_scores = topk_scores[mask]
            
            # Move to CPU and convert to list
            row_indices = row_indices.cpu().tolist()
            topk_indices = topk_indices.cpu().tolist()
            topk_scores = topk_scores.cpu().tolist()
            
            # Append edges
            all_edges.extend(zip(row_indices, topk_indices, topk_scores))

    # Add edges to the graph
    print("Adding edges to the graph...")
    G.add_weighted_edges_from(all_edges)

    return G

def apply_infomap(G):
    """
    在相似度图上应用 Infomap 进行社区检测。

    :param G: NetworkX 图
    :return: dict，sentence_id 到 topic_id 的映射, dict，topic_id 到 sentence_id 列表的映射
    """
    print("Running Infomap...")
    im = Infomap()
    for edge in tqdm(G.edges(data=True), desc='Adding edges to Infomap'):
        im.addLink(edge[0], edge[1], edge[2]['weight'])

    im.run()

    # 获取社区分配
    sid_to_topic = {}
    for node in im.nodes:
        sid_to_topic[node.node_id] = node.module_id

    # 构建 topic_to_sid
    topic_to_sid = defaultdict(list)
    for sid, topic in sid_to_topic.items():
        topic_to_sid[topic].append(sid)

    return sid_to_topic, topic_to_sid


def filter_small_topics(sid_to_topic, topic_to_sid, min_sentences=5):
    """
    过滤包含句子数量少于 min_sentences 的主题。

    :param sid_to_topic: dict，sentence_id 到 topic_id 的映射
    :param topic_to_sid: dict，topic_id 到 sentence_id 列表的映射
    :param min_sentences: 最小句子数量阈值
    :return: 过滤后的 sid_to_topic 和 topic_to_sid
    """
    print(f"Filtering topics with fewer than {min_sentences} sentences...")
    filtered_topic_to_sid = {topic: sids for topic, sids in topic_to_sid.items() if len(sids) >= min_sentences}
    filtered_sid_to_topic = {sid: topic for sid, topic in sid_to_topic.items() if topic in filtered_topic_to_sid}

    # 重新构建 topic_to_sid 以确保主题 ID 连续
    old_to_new_topic = {old: new for new, old in enumerate(filtered_topic_to_sid.keys())}
    final_topic_to_sid = defaultdict(list)
    final_sid_to_topic = {}
    for old_topic, sids in filtered_topic_to_sid.items():
        new_topic = old_to_new_topic[old_topic]
        final_topic_to_sid[new_topic].extend(sids)
        for sid in sids:
            final_sid_to_topic[sid] = new_topic

    return final_sid_to_topic, final_topic_to_sid


def build_topic_and_sentence_mapping(sid_to_topic, topic_to_sid):
    """构建 topic_and_sentence.pkl 文件的映射关系"""
    mapping = {
        'sid_to_topic': sid_to_topic,
        'topic_to_sid': topic_to_sid
    }
    return mapping


def build_ui_to_review_id(df):
    """构建 ui_to_review_id.pkl 文件的映射关系"""
    print("Building UI to Review ID mapping...")
    ui_to_review_id = {}
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Mapping UI to Review ID'):
        ui = (row['user_id'], row['item_id']) 
        review_id = int(row['review_id'])
        ui_to_review_id[ui] = review_id
    return ui_to_review_id


def build_ui_to_sentence_id(df, sentences_df):
    """构建 ui_to_sentence_id.pkl 文件的映射关系"""
    print("Building UI to Sentence ID mapping...")
    ui_to_sentence_id = defaultdict(list)
    for _, row in tqdm(sentences_df.iterrows(), total=sentences_df.shape[0], desc='Mapping UI to Sentence IDs'):
        ui = (row['user_id'], row['item_id'])   # 现在是整数类型
        sid = int(row['sentence_id'])
        ui_to_sentence_id[ui].append(sid)
    return ui_to_sentence_id


def save_pickle(data, file_path):
    """保存数据为 pickle 文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Saved {file_path}')


if __name__ == '__main__':
    save_sentence_feat(args)
