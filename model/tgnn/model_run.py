# -*- coding: utf-8 -*-
# 把 MLPPredictor 和 SentenceRetrieval 结合到一起.

import argparse
import numpy as np
from abc import ABC
import os
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from rhg_data import GraphData
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from tqdm import tqdm
from util import get_logger, args_to_dict, args_to_str
from collections import defaultdict

def config():
    parser = argparse.ArgumentParser(description='GCMC2')
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('-dn', '--dataset_name', type=str)
    parser.add_argument('-dp', '--dataset_path', type=str, help='raw dataset file path')
    parser.add_argument('--model_save_path', type=str, help='The model saving path')
    parser.add_argument('--review_feat_size', type=int, default=128)

    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=float, default=10000)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--train_min_lr', type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=8)
    parser.add_argument('--train_early_stopping_patience', type=int, default=10)
    parser.add_argument('--train_classification', type=bool, default=True)

    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--ed_alpha', type=float, default=.1)

    args = parser.parse_args()
    args.model_short_name = 'RHGC4'

    # args.dataset_name = 'Digital_Music_5'
    # args.dataset_path = '/home/d1/shuaijie/data/Digital_Music_5/Digital_Music_5.json'
    # args.review_feat_size = 64
    # args.gcn_dropout = 0.7
    # args.device = 1
    # args.num_layers = 2
    # args.batch_size = 10000

    # args.dataset_name = 'Toys_and_Games_5'
    # args.dataset_path = '/home/d1/shuaijie/data/Toys_and_Games_5/Toys_and_Games_5.json'
    # args.review_feat_size = 64
    # args.gcn_dropout = 0.8
    # args.ed_alpha = 0.1
    # args.device = 0
    # args.num_layers = 2
    # args.batch_size = 10000
    # args.epoch = 2000

    # args.dataset_name = 'Sports_and_Outdoors_5'
    # args.dataset_path = '/home/d1/shuaijie/data/Sports_and_Outdoors_5/Sports_and_Outdoors_5.json'

    args.dataset_name = 'Musical_Instruments'
    args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output/filtered_Musical_Instruments_output.jsonl'
    args.dataset_name = 'Musical_Instruments_aspect'
    args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output_aspect/filtered_Musical_Instruments_output.jsonl'
 #   args.dataset_name = 'Industrial_and_Scientific'
 #   args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Industrial_and_Scientific_output/filtered_Industrial_and_Scientific_output.jsonl' # 修改为您的 JSONL 文件路径
 #   args.dataset_name = 'yelp_reviews'
 #   args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_yelp_reviews_output/filtered_yelp_restaurant_reviews_output.jsonl' # 修改为您的 JSONL 文件路径
    args.dataset_name = 'Industrial_and_Scientific_raw_part2'
    args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Industrial_and_Scientific_aspect_raw_10-50/filtered_Industrial_and_Scientific_aspect_raw_10-50.jsonl' # 修改为您的 JSONL 文件路径


    args.gcn_dropout = 0.8
    args.ed_alpha = 2.0
    args.device = 5
    args.num_layers = 2
    args.batch_size = 1000

    # args.dataset_name = 'Health_and_Personal_Care_5'
    # args.dataset_path = '/home/d1/shuaijie/data/Health_and_Personal_Care_5/Health_and_Personal_Care_5.json'
    # args.gcn_dropout = 0.7
    # args.device = 0
    # args.epoch = 200

    # args.dataset_name = 'Yelp2013'
    # args.dataset_path = '/home/d1/shuaijie/data/yelp-recsys-2013/yelp2013.json'
    # args.gcn_dropout = 0.7
    # args.device = 1
    # args.epoch = 800
    # args.num_layers = 2

    # args.dataset_name = 'Yelp1'
    # args.dataset_path = '/home/d1/shuaijie/data/yelp1/yelp2013.json'
    # args.gcn_dropout = 0.8
    # args.device = 1
    # args.epoch = 300

    # args.device = torch.device(args.device) if args.device >= 0 else torch.device('cpu')
    args.device = f"cuda:{args.device}" if args.device >= 0 else 'cpu'

    # configure save_fir to save all the info
    args.model_save_path = f'model_save/{args.dataset_name}/{args.model_short_name}_layers_{args.num_layers}.pt'
    if not os.path.isdir(f'model_save/{args.dataset_name}'):
        os.makedirs(f'model_save/{args.dataset_name}')

    args.gcn_out_units = args.review_feat_size

    return args


def reset_parameters(model):
    em_set = set(['review_embedding.weight', 'sentence_embedding.weight'])

    for n, p in model.named_parameters():
        if n in em_set:
            continue
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)


def format_dict_to_str(data_dict):
    result = []
    for k, v in data_dict.items():
        result.append(f'{k}: {v:>.4f}')
    return ', '.join(result)


class GCMCGraphConv(nn.Module, ABC):

    def __init__(self, \
                 feature_size, \
                 review_embedding, \
                 add_embedding_mapping=False, \
		 add_review=False, \
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()

        self.embedding_mapping = nn.Linear(feature_size, feature_size) if add_embedding_mapping else None
        self.prob_score = nn.Linear(128, 1, bias=False)
        self.review_embedding = review_embedding
        if add_review:
            self.review_w = nn.Sequential(
                nn.Linear(128, feature_size, bias=False), 
                nn.GELU(),
                nn.Linear(feature_size, feature_size, bias=False), 
                nn.GELU(),
                nn.Linear(feature_size, feature_size, bias=False), 
            )
            self.review_score = nn.Linear(128, 1, bias=False)
        else:
            self.review_w = None
            self.review_score = None
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(feature_size, feature_size) 

    def get_review_feature(self, rid):
        num_embeddings = self.review_embedding.num_embeddings
        max_rid = rid.max().item()
        assert torch.all(rid < num_embeddings), f"存在 review_id 超出范围: 最大值 {rid.max().item()}，num_embeddings {num_embeddings}"
    
        review_feat = self.review_embedding(rid)
        return review_feat

    def forward(self, graph, feat):

        with graph.local_scope():

            graph.srcdata['h'] = self.embedding_mapping(feat) if self.embedding_mapping else feat 

            review_feat = self.get_review_feature(graph.edata['review_id'])
            graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))
	    
            if self.review_w is not None:
                graph.edata['ra'] = torch.sigmoid(self.review_score(review_feat))
                graph.edata['rf'] = self.review_w(review_feat)
                graph.update_all(lambda edges: {'m': (edges.src['h'] * edges.data['pa']
                                                      + edges.data['rf'] * edges.data['ra'])
                                                     * self.dropout(edges.src['cj'])},
                                 fn.sum(msg='m', out='h'))

            else:
                graph.update_all(lambda edges: {'m': edges.src['h'] * edges.data['pa']
                                                     * self.dropout(edges.src['cj'])},
                                 fn.sum(msg='m', out='h'))

            rst = graph.dstdata['h']
            rst = rst * graph.dstdata['ci']
            rst = self.linear(rst)

        return rst 


class MultiLayerHeteroGraphConv(nn.Module):

    def __init__(self, rating_values, review_embedding, user_size, item_size, msg_units, num_layers, aggregate='sum', dropout_rate=0.0):
        super(MultiLayerHeteroGraphConv, self).__init__()
        
        assert num_layers > 0, "The numbder of conv layers must have at least one!"
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        rating_values = [str(r) for r in rating_values]
        self.rating_values = rating_values

        self.user_embedding = nn.Parameter(torch.Tensor(user_size, msg_units))
        self.item_embedding = nn.Parameter(torch.Tensor(item_size, msg_units))

        sub_conv = nn.ModuleDict()

        for l in range(num_layers):
            sub_conv = {}
            for rating in rating_values:

                rating = str(rating)
                rev_rating = f'rev-{rating}'
                sub_conv[rating] = GCMCGraphConv(msg_units, \
                                                 review_embedding, \
						 add_embedding_mapping = l == 0, \
						 add_review = l == (num_layers - 1), \
						 dropout_rate=dropout_rate)
                sub_conv[rev_rating] = GCMCGraphConv(msg_units, 
                                                     review_embedding, \
						     add_embedding_mapping = l == 0, \
						     add_review = l == (num_layers - 1), \
						     dropout_rate=dropout_rate)

            self.conv_layers.append(nn.ModuleDict(sub_conv))
        
        self.ufc = nn.Linear(msg_units, msg_units)
        self.ifc = nn.Linear(msg_units, msg_units)
        self.dropout = nn.Dropout(0.5)
        self.agg_act = nn.GELU()
        

    def forward(self, input_nodes, encoder_blocks):
        
        user_outputs = []
        item_outputs = []

        # first layer
        for l in range(len(self.conv_layers)):
            u_layer_output = dict()
            m_layer_output = dict()

            block = encoder_blocks[l]
            conv_layer = self.conv_layers[l]

            for rating in self.rating_values:

                if l == 0: 
                    i_o = conv_layer[rating](block['user', rating, 'item'], 
                                             self.user_embedding[input_nodes['user']])
                    u_o = conv_layer[f'rev-{rating}'](block['item', f'rev-{rating}', 'user'], 
                                                      self.item_embedding[input_nodes['item']])
                else:
                    _u_feats = user_outputs[-1][rating]
                    _i_feats = item_outputs[-1][rating]

                    i_o = conv_layer[rating](block['user', rating, 'item'], _u_feats)
                    u_o = conv_layer[f'rev-{rating}'](block['item', f'rev-{rating}', 'user'], _i_feats)

                m_layer_output[rating] = i_o
                u_layer_output[rating] = u_o

            user_outputs.append(u_layer_output)
            item_outputs.append(m_layer_output)

        user_outputs = sum(list(user_outputs[-1].values()))
        item_outputs = sum(list(item_outputs[-1].values()))
        # user_outputs = user_outputs.sum(1)
        user_outputs = self.agg_act(user_outputs)
        user_outputs = self.dropout(user_outputs)
        user_outputs = self.ufc(user_outputs)
        # item_outputs = item_outputs.sum(1)
        item_outputs = self.agg_act(item_outputs)
        item_outputs = self.dropout(item_outputs)
        item_outputs = self.ifc(item_outputs)

        return user_outputs, item_outputs


class ContrastLoss(nn.Module, ABC):

    def __init__(self, h_size, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_size, h_size))
        torch.nn.init.xavier_uniform_(self.w.data)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # y += torch.zeros_like(y).normal_(0, 0.01)

        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]

        # y += y_sim
        neg_scores = (y_neg @ self.w * x).sum(1)
        neg_labels = neg_scores.new_zeros(neg_scores.shape)
        neg_loss = self.bce_loss(neg_scores, neg_labels)

        scores = (y @ self.w * x).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        loss = pos_loss + neg_loss
        return loss

    def measure_sim(self, x, y):
        if len(y.shape) < 3:
            scores = (y @ self.w * x).sum(1).sigmoid()
        else:
            scores = torch.einsum('bld,bd->bl', y @ self.w, x).sigmoid()
        return scores


class TopicGraphEncoder(nn.Module):

    def __init__(self, sentence_embedding, topic_size, feature_size):
        super().__init__()

        self.sentence_embedding = sentence_embedding

        self.sentence_w = nn.Sequential(
            nn.Linear(128, feature_size, bias=False), 
            # Affine(feature_size),
            nn.GELU(),
            nn.Linear(feature_size, feature_size, bias=False), 
            # Affine(feature_size),
            nn.GELU(),
            nn.Linear(feature_size, feature_size, bias=False), 
        )
        self.gelu = nn.GELU()

        self.sentence_w1 = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.sentence_score_w = nn.Parameter(torch.Tensor(topic_size, feature_size))
        # self.sentence_w2 = nn.Parameter(torch.Tensor(topic_size, feature_size, feature_size))

        self.sentence_linear = nn.Linear(feature_size, feature_size)

        self.topic_user_linear = nn.Linear(feature_size, feature_size)
        self.topic_item_linear = nn.Linear(feature_size, feature_size)
        self.topic_user_w = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.topic_item_w = nn.Parameter(torch.Tensor(topic_size, feature_size))

        self.dropout = nn.Dropout(0.5)

    def sentence_to_topic(self, graph, sentence_id):
        sent_feat = self.sentence_embedding(sentence_id)
        # sent_feat = self.sentence_w(sent_feat)

        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.sentence_w1[stid] * sent_feat
        # graph.srcdata['attn_score'] = (self.sentence_score_w[stid] * sent_feat).sum(-1, keepdim=True)

        with graph.local_scope():

            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            
            # graph.apply_edges(lambda e: {'attn_score': e.src['attn_score']})
            graph.edata['attn_score'] = edge_softmax(graph, graph.edata['attn_score'])

            # message passing with attention
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            
            result = graph.dstdata['h']

        result = self.sentence_linear(result)
        return result

    def topic_to_user_item(self, graphs, topic_feat):

        graph = graphs[('topic', 'topic_to_user', 'user')]
        # graph.srcdata['h'] = topic_feat
        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.gelu(topic_feat * self.topic_user_w[stid])

        with graph.local_scope():

            # calculate attention weight
            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            e_attn = graph.edata['attn_score']
            graph.edata['attn_score'] = edge_softmax(graph, e_attn)

            # message passing with attention
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            
            user_feat = graph.dstdata['h']

        user_feat = self.topic_user_linear(user_feat)
        
        # item
        graph = graphs[('topic', 'topic_to_item', 'item')]
        # graph.srcdata['h'] = topic_feat
        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.gelu(topic_feat * self.topic_item_w[stid])

        with graph.local_scope():

            # calculate attention weight
            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            e_attn = graph.edata['attn_score']
            graph.edata['attn_score'] = edge_softmax(graph, e_attn)

            # message passing with attention
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            
            item_feat = graph.dstdata['h']

        item_feat = self.topic_item_linear(item_feat)

        return user_feat, item_feat

    def forward(self, input_nodes, encoder_blocks):
        topic_embedding = self.sentence_to_topic(encoder_blocks[0][('sentence', 'sentence_to_topic', 'topic')], \
                                                 input_nodes['sentence'])
        uo, io = self.topic_to_user_item(encoder_blocks[1], topic_embedding)
        return uo, io


class SentenceRetrival(nn.Module):

    def __init__(self,
                 in_units,
                 num_classes,
                 review_embedding,
                 sentence_embedding,
                 dropout_rate=0.0):
        super(SentenceRetrival, self).__init__()

        self.sentence_embedding = sentence_embedding
        self.review_embedding = review_embedding
        print(f"Sentence Embedding - num_embeddings: {self.sentence_embedding.num_embeddings}, embedding_dim: {self.sentence_embedding.embedding_dim}")

        self.rating_linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.topic_linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.rating_predictor = nn.Linear(in_units, num_classes, bias=False)
        self.contrast_loss = ContrastLoss(in_units, 128)

    def get_review_feature(self, sid):
        # sid: bs * k
        length = (sid > 0).float().sum(dim=-1, keepdim=True) + 1e-9
        review_feat = self.sentence_embedding(sid).sum(dim=-2)
        review_feat = review_feat / length
        return review_feat

    def calc_ranking_loss(self, edges):

        # rating
        rh = self.rating_linear(torch.cat([edges.src['rf'], edges.dst['rf']], dim=1))
        # review_feat = self.review_embedding(edges.data['review_id'])
        pr = self.rating_predictor(rh)
        # mi_score = self.contrast_loss(rh, review_feat)

        # topic
        th = self.topic_linear(torch.cat([edges.src['tf'], edges.dst['tf']], dim=1))
        th = th + rh
        pos_sid = edges.data['sentence_id']
        neg_sid = torch.randint(1, self.sentence_embedding.weight.shape[0], \
                                pos_sid.shape, 
                                device=pos_sid.device)

        pos_review = self.get_review_feature(pos_sid)
        neg_review = self.get_review_feature(neg_sid)

        pos_score = (th * pos_review).sum(1)
        neg_score = (th * neg_review).sum(1)
        loss = - (pos_score - neg_score).sigmoid().log()

        # return {'p_ratings': pr, 'mi_score': mi_score, 'ranking_loss': loss}
        return {'p_ratings': pr, 'mi_score': loss, 'ranking_loss': loss}

    def predict_rating(self, graph, urf, irf, utf, itf):
        # graph.nodes['item'].data['th'] = itf
        # graph.nodes['user'].data['th'] = utf
        graph.nodes['item'].data['rf'] = irf
        graph.nodes['user'].data['rf'] = urf

        _rating_predic_func = lambda e: {'p': self.rating_predictor(
            self.rating_linear(torch.cat([e.src['rf'], e.dst['rf']], dim=1))
        )}

        with graph.local_scope():
            graph.apply_edges(_rating_predic_func)
            pr = graph.edata['p']
        return pr
        
    def forward(self, graph, urf, irf, utf, itf):
        graph.nodes['user'].data['rf'] = urf
        graph.nodes['item'].data['rf'] = irf
        graph.nodes['user'].data['tf'] = utf
        graph.nodes['item'].data['tf'] = itf

        with graph.local_scope():
            graph.apply_edges(self.calc_ranking_loss)
            pr, mi_score, ranking_loss = graph.edata['p_ratings'], graph.edata['mi_score'], graph.edata['ranking_loss']
        return pr, mi_score.mean(), ranking_loss.mean()

    def measure_sim(self, interaction_feat, sid_list):
        # bs * dim, bs * k
        min_sid = torch.min(sid_list).item()
        max_sid = torch.max(sid_list).item()
        num_embeddings = self.sentence_embedding.num_embeddings

#        print(f"sid_list - min: {min_sid}, max: {max_sid}, num_embeddings: {num_embeddings}")

          # 添加断言确保索引合法
        assert min_sid >= 0, f"sid_list contains negative indices: min_sid={min_sid}"
        assert max_sid < num_embeddings, f"sid_list contains indices >= num_embeddings: max_sid={max_sid}, num_embeddings={num_embeddings}"


        sent_feat = self.sentence_embedding(sid_list)  # bs * k * dim
        scores = torch.einsum('bd,bkd->bk', interaction_feat, sent_feat)
        return scores

    # 分 batch 计算 ranking method
    @staticmethod
    def _rank_batch(_h, _cand, _trues, _measure_func, topk):
        """
        _h: 交互表征
        _cand: 可能的sentence id list
        _trues: 真实的 sentence id list
        _measure_func: 
        """
        _cand_mask = (_cand > 0).float()
        _ml = _cand_mask.int().sum(dim=1).max()
        _cand = _cand[:, :_ml]
        _cand_mask = _cand_mask[:, :_ml]
        _scores = _measure_func(_h, _cand)
        _, _topk_idx = torch.topk(_scores, k=topk, dim=-1)
        _topk_items = torch.gather(_cand, 1, _topk_idx)
        _topk_items = _topk_items.cpu().numpy()
        _trues = _trues.cpu().numpy()
        # import pdb; pdb.set_trace()
        return calc_ranking_metrics(_topk_items, _trues)

    @ torch.no_grad()
    def get_ranking_scores(self, graph, user_feat, item_feat, topk=5):
        graph.nodes['item'].data['th'] = item_feat
        graph.nodes['user'].data['th'] = user_feat

        def _get(edges): 
            h = self.topic_linear(torch.cat([edges.src['th'], edges.dst['th']], dim=1))
            cand_sent = edges.dst['candidate_sentence_id']
            return {'th': h, 'cand_sid': cand_sent}

        graph.apply_edges(_get)

        h = graph.edata['th']
        true_sents = graph.edata['sentence_id']
        cand_sents = graph.edata['cand_sid']

        rank_list = []
        _bs = 2000
        for i in range(0, h.shape[0], _bs):
            _sent_scores = self._rank_batch(h[i: i + _bs], \
                                            cand_sents[i: i + _bs],
                                            true_sents[i: i + _bs], \
                                            self.measure_sim, \
                                            topk=topk)
            rank_list.append(_sent_scores)

        result = {k: sum([list(_rl[k]) for _rl in rank_list], [])
                  for k in rank_list[0].keys()}
        # result = {k: np.mean(v) for k, v in result.items()}
        return result

	    
def calc_ranking_metrics(topk_items, true_list):
    precision, recall = precision_recall_score(topk_items, true_list)
    f1 = [ 2 * p * r / (p + r) if p + r > 0. else 0. for p, r in zip(precision, recall) ]
    ndcg = ndcg_score(topk_items, true_list)
    
    return {'Pre': precision, \
            'Rec': recall, \
            'F1': f1, \
            'nDCG': ndcg}
    

def precision_recall_score(predicts, trues):
    
    def pr_each(ps, ts):
        ps = ps[ps > 0]
        ts = ts[ts > 0]
        if len(ts) < 1 or len(ps) < 1:  # some reviews are empty
            return 0., 0.
        inter = np.intersect1d(ps, ts)
        return len(inter) / len(ps), len(inter) / len(ts)
    
    prs, rcs = zip(*[pr_each(predicts[i], trues[i]) for i in range(len(predicts))])
    return prs, rcs


def ndcg_score(predicts, trues):
    
    def _ndcg(ps, ts):
        ps = ps[ps > 0]
        ts = ts[ts > 0]
        # if len(ts) < 1:
        #     return 0.
        if len(ts) < 1 or len(ps) < 1:  # some reviews are empty
            return 0.
        isin = np.isin(ps, ts)
        if isin.sum() == 0.:
            return 0.
        dcg = isin / np.log2(np.arange(2, len(isin) + 2))
        # idcg = 1 / np.log2(np.arange(2, len(isin) + 2))
        idcg = np.sort(isin)[::-1] / np.log2(np.arange(2, len(isin) + 2))
        return np.sum(dcg) / np.sum(idcg)
    
    return [_ndcg(predicts[i], trues[i]) for i in range(len(predicts))]


class Net(nn.Module):

    def __init__(self, review_embedding, sentence_embedding, params):
        super(Net, self).__init__()

        self.sentence_embedding = sentence_embedding# nn.Embedding.from_pretrained(sentence_embedding)
        self.review_embedding = nn.Embedding.from_pretrained(review_embedding)
        self.rating_encoder = MultiLayerHeteroGraphConv(params.rating_values, \
                                                        self.review_embedding, \
                                                        params.user_size, \
                                                        params.item_size, \
                                                        params.gcn_out_units, \
                                                        params.num_layers, \
                                                        dropout_rate=params.gcn_dropout)

        self.topic_encoder = TopicGraphEncoder(self.sentence_embedding, params.global_topic_size, params.gcn_out_units)
        self.topic_decoder = SentenceRetrival(params.gcn_out_units, 5, self.review_embedding, self.sentence_embedding)
        
        self.register_buffer('rating_values', torch.FloatTensor(params.rating_values).view(1, -1))

        reset_parameters(self)

        self.rating_loss_net = nn.CrossEntropyLoss()

    def state_dict(self):
        # exclude review embedding
        sd = super().state_dict()
        pop_keys = []
        for k in sd.keys():
            if 'review_embedding' in k or 'sentence_embedding' in k:
                pop_keys.append(k)
        for k in pop_keys:
            sd.pop(k)
        return sd

    def predict_rating(self, input_nodes, encoder_blocks, decoder_graph):
        user_feat, item_feat = self.rating_encoder(input_nodes, encoder_blocks)
        predicts = self.topic_decoder.predict_rating(decoder_graph, user_feat, item_feat, None, None)
        predicts = self.predicts_to_ratings(predicts)
        return predicts

    def calc_loss(self, \
                  rating_input_nodes, \
                  rating_encoder_blocks, \
                  topic_input_nodes, \
                  topic_encoder_blocks, \
                  decoder_graph):
        self.train()

        urf, irf = self.rating_encoder(rating_input_nodes, rating_encoder_blocks)
        utf, itf = self.topic_encoder(topic_input_nodes, topic_encoder_blocks)

        predicts, ed_mi, ranking_loss = self.topic_decoder(decoder_graph, \
                                                           urf, irf, \
                                                           # utf, itf)
                                                           utf + urf, itf + itf)

        rating_loss = self.rating_loss_net(predicts, decoder_graph.edata['label'])

        return rating_loss, ed_mi, ranking_loss

    @torch.no_grad()
    def evaluate_rating(self, dataloader, etype='valid'):
        device = self.review_embedding.weight.device
        rmse_list = []
        mae_list = []  # List to store MAE values
        # 用于保存每个评分组（1,2,3,4,5）的误差平方列表
        group_errors = defaultdict(list)

        self.eval()
        for input_nodes, edge_subgraph, blocks in dataloader:
            input_nodes = {k: v.to(device) for k, v in input_nodes.items()}
            edge_subgraph = edge_subgraph[etype].to(device)
            blocks = [b.to(device) for b in blocks]

            p_ratings = self.predict_rating(input_nodes, blocks, edge_subgraph)
            true_ratings = edge_subgraph.edata['rating']

            # 遍历每个样本，将误差平方按照真实评分分组
            for pred, true in zip(p_ratings.cpu().tolist(), true_ratings.cpu().tolist()):
                error_sq = (pred - true) ** 2
                group = int(true)  # 假设真实评分为 1,2,3,4,5
                group_errors[group].append(error_sq)
            
            rmse = p_ratings - edge_subgraph.edata['rating']
            mae = torch.abs(rmse)  # Absolute error for MAE

            rmse_list.extend(rmse.cpu().tolist())
            mae_list.extend(mae.cpu().tolist())

        # 对每个评分组计算均值（即 MSE）
        group_mse = {group: np.mean(errors) for group, errors in group_errors.items()}
        # 输出各评分组的 MSE
        print("各评分组（1～5）的均方误差（MSE）：")
        for group in sorted(group_mse.keys()):
            print("评分 {}: MSE = {:.4f}".format(group, group_mse[group]))

        rmse_value = np.sqrt(np.power(np.array(rmse_list), 2).mean())
        mae_value = np.array(mae_list).mean()  # Calculate mean of absolute errors
        mse_value = np.power(np.array(rmse_list), 2).mean()
        return rmse_value, mae_value, mse_value
    
    @torch.no_grad()
    def evaluate_sentence_ranking(self, dataloader, raw_graph, sampler, etype='valid', topk=5):
        device = self.review_embedding.weight.device
        # group_scores 用于保存每个评分组下的各指标列表
        group_scores = defaultdict(lambda: defaultdict(list))
        scores_list = []
        for rating_input_nodes, decoder_graph, rating_encoder_blocks in dataloader:
            input_nodes, _, blocks = sampler.sample(raw_graph, \
                                                    {'user': decoder_graph.nodes['user'].data['_ID'], \
                                                     'item': decoder_graph.nodes['item'].data['_ID']})

            rating_input_nodes = {k: v.to(device) for k, v in rating_input_nodes.items()}
            input_nodes = {k: v.to(device) for k, v in input_nodes.items()}
            blocks = [b.to(device) for b in blocks]
            rating_encoder_blocks = [b.to(device) for b in rating_encoder_blocks]
            decoder_graph = decoder_graph[etype].to(device)

            # 获取每个样本的真实评分（假设取值为 1～5）
            ratings = decoder_graph.edata['rating'].cpu().tolist()
            
            urf, irf = self.rating_encoder(rating_input_nodes, rating_encoder_blocks)
            utf, itf = self.topic_encoder(input_nodes, blocks)
            ranking_scores = self.topic_decoder.get_ranking_scores(decoder_graph, \
                                                                   utf + urf, \
                                                                   # itf + itf, \
                                                                   itf + irf, \
                                                                   topk)
            scores_list.append(ranking_scores)
                    # 将每个样本的各指标按照真实评分分组
            for idx, rating in enumerate(ratings):
                group = int(rating)  # 取值 1～5
                for metric, values in ranking_scores.items():
                    group_scores[group][metric].append(values[idx])

            # 对每个评分组下各指标计算平均值
        group_metrics = {}
        for group, metrics in group_scores.items():
            group_metrics[group] = {metric: np.mean(vals) for metric, vals in metrics.items()}
        
        # 输出每个评分组的排序指标
        print("各评分组（1～5）的排序指标：")
        for group in sorted(group_metrics.keys()):
            metrics = group_metrics[group]
            print("评分组 {}: Pre = {:.4f}, Rec = {:.4f}, F1 = {:.4f}, nDCG = {:.4f}".format(
                group, metrics.get('Pre', 0), metrics.get('Rec', 0), metrics.get('F1', 0), metrics.get('nDCG', 0)
            ))
        
        scores_list = {k: sum([list(_rl[k]) for _rl in scores_list], [])
                       for k in scores_list[0].keys()}
        scores_list = {k: np.mean(v) for k, v in scores_list.items()}
        return scores_list


    def predicts_to_ratings(self, predicts):
        if len(predicts) < 2:
            return predicts
        else:
            return (torch.softmax(predicts, dim=1) * self.rating_values).sum(dim=1)
        

def train(params):

    # global logger

    # save_dir, code_dir =  make_trainging_log_dir(base_dir='logs/', \
    #     					 data_name=params.dataset_name, \
    #     					 model_name=params.model_short_name)
    # params.model_save_path = f"{save_dir}/model.pkl"

    # copy_py_files('./', code_dir)

    # logger = get_logger(params.model_short_name, f"{save_dir}/training.log")

    # global tb_logger
    # tb_logger = get_tensorboard_writer(f"{save_dir}/tb")

    global logger

    logger = get_logger(params.model_short_name, None)

    logger.info(f"Parameters:\n{args_to_str(params)}")

    dataset = GraphData(params.dataset_name,
                        params.dataset_path) 
                        # device='cpu')

    # 获取所有数据集中的句子ID
    train_sentence_ids = dataset.train_sentence_ids
    valid_sentence_ids = dataset.valid_sentence_ids
    test_sentence_ids = dataset.test_sentence_ids

    params.user_size = dataset.user_size
    params.item_size = dataset.item_size
    params.rating_values = dataset.possible_rating_values

    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1


    # 计算所有句子ID的最大值
    all_sentence_ids = torch.cat([train_sentence_ids, valid_sentence_ids, test_sentence_ids])
    max_sentence_id = torch.max(all_sentence_ids).item()
    current_num_embeddings = dataset.sentence_embedding.num_embeddings

    print(f"All sentence IDs - min: {torch.min(all_sentence_ids).item()}, max: {max_sentence_id}")
    print(f"Current sentence_embedding - num_embeddings: {current_num_embeddings}")

    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    net = net.to(params.device)

    learning_rate = params.train_lr
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    logger.info("Loading network finished ...\n")

    train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=params.batch_size, num_layers=params.num_layers)
    graph = dataset.graph
    topic_sampler = dataset.get_topic_sentence_sampler()

    best_valid_rmse = np.inf
    best_test_rmse = np.inf
    best_test_mae = np.inf
    best_test_mse = np.inf

    no_better_valid = 0
    best_iter = -1
    
    # logger.info('Valid -' + format_dict_to_str(net.evaluate_sentence_ranking(valid_dataloader, graph, topic_sampler, etype='valid')))
    logger.info('Test - '+ format_dict_to_str(net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')))

    logger.info("Start training ...")
    for iter_idx in range(1, params.epoch):
        net.train()

        pbar = tqdm(train_dataloader)
        # pbar = train_dataloader
        train_rmse = []
        train_mi = []
        for rating_input_nodes, edge_subgraph, rating_blocks in pbar:
            # input_nodes: 表示计算 edge_subgraph 所需要的节点，
            # edge_subgraph: sample 出来的 graph 
            # rating_blocks:包含了每个GNN层要计算哪些节点表示作为输出，要将哪些节点表示作为输入，以及来自输入节点的表示如何传播到输出节点。
            
            # decoder_graph.nodes['item'].data['_ID']
            topic_input_nodes, _, topic_blocks = topic_sampler.sample(graph, \
                                                                      {'user': edge_subgraph.nodes['user'].data['_ID'], \
                                                                       'item': edge_subgraph.nodes['item'].data['_ID']})

            rating_input_nodes = {k: v.to(params.device) for k, v in rating_input_nodes.items()}
            topic_input_nodes = {k: v.to(params.device) for k, v in topic_input_nodes.items()}
            edge_subgraph = edge_subgraph['train'].to(params.device)
            rating_blocks = [b.to(params.device) for b in rating_blocks]
            topic_blocks = [b.to(params.device) for b in topic_blocks]

            r_loss, mi_score, ranking_loss = net.calc_loss(rating_input_nodes, rating_blocks, topic_input_nodes, topic_blocks, edge_subgraph)
            # loss = params.ed_alpha * mi_score + r_loss + ranking_loss
            loss = params.ed_alpha * mi_score + r_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
            optimizer.step()

            # print(f"ranking:{ranking_loss.item():.4f}")
            pbar.set_description(f"train_loss={r_loss:.4f}, MI={mi_score:.2f}, ranking={ranking_loss.item():.4f}")
            
            # with torch.no_grad():
            #     predict_ratings = net.predicts_to_ratings(predicts)
            #     rmse = predict_ratings - edge_subgraph.edata['rating']
            #     train_rmse.extend(rmse.cpu().tolist())
            #     train_mi.append(mi_score.cpu().item())

        # train_rmse = np.sqrt(np.power(np.array(train_rmse), 2).mean())
        # train_mi = np.mean(train_mi)
        train_rmse = r_loss.item()
        train_mi = mi_score.item()

        valid_rmse,_,_ = net.evaluate_rating(valid_dataloader, etype='valid')
        logging_str = f"Epoch={iter_idx:>3d}, " \
                      f"Train_Loss={train_rmse:.4f}, MI={train_mi:.2f}, Valid_RMSE={valid_rmse:.4f}, "

        # tb_logger.add_scalar('Train_RMSE', train_rmse, iter_idx)
        # tb_logger.add_scalar('Valid_RMSE', valid_rmse, iter_idx)

        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            no_better_valid = 0
            best_iter = iter_idx
            # test_rmse = evaluate(params.device, net, test_dataloader, 'test')
            test_rmse, test_mae, test_mse = net.evaluate_rating(test_dataloader, etype='test')
            best_test_rmse = test_rmse
            best_test_mae = test_mae
            best_test_mse = test_mse

            logging_str += f'Test RMSE={test_rmse:.4f}'
            # tb_logger.add_scalar('Test_RMSE', valid_rmse, iter_idx)
            torch.save(net.state_dict(), params.model_save_path)
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience and learning_rate <= params.train_min_lr:
                logger.info("Early stopping threshold reached. Stop training.")
                break
            if no_better_valid > params.train_decay_patience:
                new_lr = max(learning_rate * params.train_lr_decay_factor, params.train_min_lr)
                if new_lr < learning_rate:
                    learning_rate = new_lr
                    logger.info("\tChange the LR to %g" % new_lr)
                    for p in optimizer.param_groups:
                        p['lr'] = learning_rate
                    no_better_valid = 0

        logger.info(logging_str)
        # print('Valid -', format_dict_to_str(net.evaluate_sentence_ranking(valid_dataloader, graph, topic_sampler, etype='valid')))
        logger.info('Test - ' + format_dict_to_str(net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')))
        
    hparam_dict = args_to_dict(params)
    key_hparam_list = ['review_feat_size', 'gcn_dropout', 'num_layers']
    hparam_dict = {k: hparam_dict[k] for k in key_hparam_list}
    # tb_logger.add_hparams(hparam_dict=hparam_dict, \
    #                       metric_dict={'Valid_RMSE': best_valid_rmse, 'Test_RMSE': best_test_rmse}, \
    #                       run_name='metric')

    logger.info(f'Best Iter Idx={best_iter}, Best Valid RMSE={best_valid_rmse:.4f}, Best Test RMSE={best_test_rmse:.4f}, Best Test MAE={best_test_mae:.4f}, Best Test MSE={best_test_mse:.4f}')
    logger.info(params.model_save_path)


def test(params):
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    # logger = get_logger(params.model_short_name, None)

    dataset = GraphData(params.dataset_name,
                        params.dataset_path) 
                        # device='cpu')

    params.user_size = dataset.user_size
    params.item_size = dataset.item_size
    params.rating_values = dataset.possible_rating_values

    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=params.batch_size, num_layers=params.num_layers)
    graph = dataset.graph
    topic_sampler = dataset.get_topic_sentence_sampler()

    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    net.load_state_dict(torch.load(params.model_save_path), strict=False)
    net = net.to(params.device)

    test_rmse,test_mae,test_mse = net.evaluate_rating(test_dataloader, etype='test')

    print(params.dataset_name)
    print(f'Test RMSE={test_rmse:.4f},Test MAE={test_mae:.4f},Test MSE={test_mse:.4f}')
    print('Pre     Rec     F1      nDCG')
    for k in [10, 50]:
        scores = net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test', topk=k)
        print('{Pre:.4f}\t{Rec:.4f}\t{F1:.4f}\t{nDCG:.4f}'.format(**scores))




def calc_bleu_metric(predict_list, true_list):
    # list of string

    b1l, b2l, b4l = [], [], []

    for p, t in zip(predict_list, true_list):
        p = p.split()
        t = t.split()
        b1 = sentence_bleu([t], p, weights=(1, 0, 0, 0))
        b2 = sentence_bleu([t], p, weights=(0.5, 0.5, 0, 0))
        b4 = sentence_bleu([t], p, weights=(0.25, 0.25, 0.25, 0.25))
        b1l.append(b1)
        b2l.append(b2)
        b4l.append(b4)

    return {'BLEU-1': np.mean(b1l), 'BLEU-2': np.mean(b2l), 'BLEU-4': np.mean(b4l)}


def calc_rouge_metric(predict_list, true_list):
    rouge = Rouge()
    predict_list = [' '.join(x) for x in predict_list]
    true_list = [' '.join(x) for x in true_list]
    rouge_scores = rouge.get_scores(predict_list, true_list, avg=True)
    rouge_scores = {k: v['f'] for k, v in rouge_scores.items()}
    return rouge_scores


if __name__ == '__main__':
    config_args = config()
    train(config_args)
    test(config_args)

