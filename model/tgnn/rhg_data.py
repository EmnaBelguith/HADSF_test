# -*- coding: utf-8 -*-

# 用 Word2Vec 词向量建模评论特征
import numpy as np
import scipy.sparse as sp
import torch
import dgl
import pickle
import os
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm
from collections import defaultdict
from load_data import load_sentiment_data
import torch.nn as nn


def load_pickle(path):
    with open(path, 'rb') as f:
            return pickle.load(f)


class GraphData(object):

    def __init__(self, dataset_name, dataset_path):
        """ 从 user-rating-item 交互中构建一张大图，训练集中的 rating 数据直接用于构造边的类型。
        每条边都有一个对应的 rating 和 label 属性。
        训练集，验证集和测试集的交互用 train, valid 和 test 作为 edge type 做标注，并加上 rating 和 label 属性。
        这样做的目的是为了方便做子图的抽样.
        """

        dir_path = f'/home/zheng/RatingTopicGraph/checkpoint/{dataset_name}/'

        self.possible_rating_values = np.arange(1, 6)
        graph_path = f'{dir_path}/hyper_graph.bin'

        if os.path.exists(graph_path):
            self.graph = load_graphs(graph_path, [0])[0][0]

            # 从加载的图中提取句子ID
            self.train_sentence_ids = self.graph['train'].edata['sentence_id']
            self.valid_sentence_ids = self.graph['valid'].edata['sentence_id']
            self.test_sentence_ids = self.graph['test'].edata['sentence_id']

        else:

            sent_train_data, sent_valid_data, sent_test_data, _, _, _ = \
                load_sentiment_data(dataset_path)

            # uir: users, items, ratings
            process_uir = lambda x: (torch.from_numpy(x['user_id'].to_numpy().astype(np.int64)), \
                                     torch.from_numpy(x['item_id'].to_numpy().astype(np.int64)), \
                                     torch.from_numpy(x['rating'].to_numpy().astype(np.float32)))

            self.train_uir = process_uir(sent_train_data)
            self.valid_uir = process_uir(sent_valid_data)
            self.test_uir = process_uir(sent_test_data)


            # self.user_size = dataset_info['user_size']
            # self.item_size = dataset_info['item_size']

            topic_info = load_pickle(f'{dir_path}/BERT-Whitening/topic_and_sentence.pkl')
            self.sid_to_topic = topic_info['sid_to_topic']
            self.topic_to_sid = topic_info['topic_to_sid']


            self.ui_to_rid = load_pickle(f'{dir_path}/BERT-Whitening/ui_to_review_id.pkl')
            self.ui_to_sid = load_pickle(f'{dir_path}/BERT-Whitening/ui_to_sentence_id.pkl')
            self.graph = self._generate_graph()
            save_graphs(graph_path, [self.graph])


        self.user_size = self.graph.num_nodes('user')
        self.item_size = self.graph.num_nodes('item')

        self.review_embedding = torch.from_numpy(np.load(f'{dir_path}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_128_whitening_review_embedding_128.npy').astype(np.float32))
        self.sentence_embedding = torch.from_numpy(np.load(f'{dir_path}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_128_whitening_sentence_embedding_128.npy').astype(np.float32))
        self.review_embedding[0] = 0
        self.sentence_embedding[0] = 0

        # TODO 
        # tmp = [0] * len(self.sid_to_topic)
        # for s, t 
        # self.sid_to_topic = torch.LongTensor()
        # 将句子嵌入转换为 nn.Embedding 实例
        self.sentence_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.sentence_embedding), freeze=False)

        # 调整嵌入层的大小以涵盖所有句子ID
        all_sentence_ids = torch.cat([self.train_sentence_ids, self.valid_sentence_ids, self.test_sentence_ids])
        max_sentence_id = torch.max(all_sentence_ids).item()
        current_num_embeddings = self.sentence_embedding.num_embeddings

        print(f"All sentence IDs - min: {torch.min(all_sentence_ids).item()}, max: {max_sentence_id}")
        print(f"Current sentence_embedding - num_embeddings: {current_num_embeddings}")

        if current_num_embeddings <= max_sentence_id:
            new_num_embeddings = max_sentence_id + 1# 确保所有sid都在范围内
            print(f"Adjusting sentence_embedding from {current_num_embeddings} to {new_num_embeddings}")

            new_embedding = nn.Embedding(new_num_embeddings, self.sentence_embedding.embedding_dim)
            with torch.no_grad():
                new_embedding.weight[:current_num_embeddings] = self.sentence_embedding.weight.data
                # 初始化新增部分，可以使用随机初始化或设为零
                new_embedding.weight[current_num_embeddings:] = torch.randn(new_num_embeddings - current_num_embeddings, self.sentence_embedding.embedding_dim)
            
            self.sentence_embedding = new_embedding
            print(f"Adjusted sentence_embedding to have num_embeddings={new_num_embeddings}")
        else:
            print("No adjustment needed for sentence_embedding.")

        # 调整 review_embedding 的大小
        # 这里需要遍历所有边类型，获取所有 'review_id' 的最大值
        # 调整 review_embedding 的大小
        max_review_id = 0
        for etype in self.graph.etypes:
            edge_data = self.graph.edges[etype].data
            if 'review_id' in edge_data:
                current_max = edge_data['review_id'].max().item()
                if current_max > max_review_id:
                    max_review_id = current_max

        current_num_review_embeddings = self.review_embedding.shape[0]

        print(f"All review IDs - max: {max_review_id}")
        print(f"Current review_embedding - num_embeddings: {current_num_review_embeddings}")

        if current_num_review_embeddings <= max_review_id:
            new_num_review_embeddings = max_review_id + 1  # 确保所有 review_id 都在范围内
            print(f"Adjusting review_embedding from {current_num_review_embeddings} to {new_num_review_embeddings}")

            # 扩展嵌入张量
            new_review_embedding = torch.cat([
                self.review_embedding,
                torch.randn(new_num_review_embeddings - current_num_review_embeddings, self.review_embedding.shape[1])
            ], dim=0)

            self.review_embedding = new_review_embedding
            print(f"Adjusted review_embedding to have num_embeddings={new_num_review_embeddings}")
        else:
            print("No adjustment needed for review_embedding.")
        

    def _generate_graph(self):

        # review id of each interaction
        train_rid_list = [self.ui_to_rid[(self.train_uir[0][x].item(), self.train_uir[1][x].item())] \
                          for x in range(self.train_uir[0].shape[0])]
        train_rid_list = torch.LongTensor(train_rid_list)
        valid_rid_list = [self.ui_to_rid[(self.valid_uir[0][x].item(), self.valid_uir[1][x].item())] \
                          for x in range(self.valid_uir[0].shape[0])]
        valid_rid_list = torch.LongTensor(valid_rid_list)
        test_rid_list = [self.ui_to_rid[(self.test_uir[0][x].item(), self.test_uir[1][x].item())] \
                          for x in range(self.test_uir[0].shape[0])]
        test_rid_list = torch.LongTensor(test_rid_list)

        # sentences id of each interaction

        ## get min sentence count in a record
        def min_count(list_of_data):
            rsc = [len(x) for x in list_of_data]
            xc = int(len(rsc) * 0.05)
            return np.array(rsc)[np.argpartition(rsc, -xc)[-xc:]].min().item()

        self.min_sentence_count = min_count(self.ui_to_sid.values())

        def _pad(sl, ml):
           sl = sl[:ml]
           if len(sl) < ml:
               sl += [0] * (ml - len(sl))
           return sl

        ## map to torch.LongTensor
        def _get_sid_list(uir):
            _sll = []
            uc = 0  # 没有对应 sid 的 ui pair 统计。
            for idx in range(uir[0].shape[0]):
                ui = (uir[0][idx].item(), uir[1][idx].item())
                if ui in self.ui_to_sid:
                    _sll.append(_pad(self.ui_to_sid[ui], self.min_sentence_count))
                else:
                    _sll.append([0] * self.min_sentence_count)
                    uc += 1
            print("no sid:")
            print(uc)
            return torch.LongTensor(_sll)

        train_sid_list = _get_sid_list(self.train_uir)
        valid_sid_list = _get_sid_list(self.valid_uir)
        test_sid_list = _get_sid_list(self.test_uir)

        # 将句子ID赋值为类的属性
        self.train_sentence_ids = train_sid_list
        self.valid_sentence_ids = valid_sid_list
        self.test_sentence_ids = test_sid_list

        # train_sid_list = [_pad(self.ui_to_sid[(self.train_uir[0][x].item(), self.train_uir[1][x].item())], \
        #                                       self.min_sentence_count) \
        #                   for x in range(self.train_uir[0].shape[0])]
        # train_sid_list = torch.LongTensor(train_sid_list)

        # valid_sid_list = [_pad(self.ui_to_sid[(self.valid_uir[0][x].item(), self.valid_uir[1][x].item())], \
        #                                       self.min_sentence_count) \
        #                   for x in range(self.valid_uir[0].shape[0])]
        # valid_sid_list = torch.LongTensor(valid_sid_list)

        # test_sid_list = [_pad(self.ui_to_sid[(self.test_uir[0][x].item(), self.test_uir[1][x].item())], \
        #                                       self.min_sentence_count) \
        #                  for x in range(self.test_uir[0].shape[0])]
        # test_sid_list = torch.LongTensor(test_sid_list)

        data_dict = dict()
        rating_to_rid = dict()
        # num_nodes_dict = {'user': self.user_size, 'item': self.item_size}
        rating_row, rating_col = self.train_uir[:2]
        for rating in self.possible_rating_values:
            ridx = np.where(self.train_uir[2] == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = str(rating)
            data_dict.update({
                ('user', str(rating), 'item'): (rrow, rcol),
                ('item', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
            rating_to_rid[rating] = train_rid_list[ridx]

        data_dict[('user', 'train', 'item')] = self.train_uir[:2]
        data_dict[('user', 'valid', 'item')] = self.valid_uir[:2]
        data_dict[('user', 'test', 'item')] = self.test_uir[:2]

        # sentence-topic-user, sentence-topic-item
        item_to_sl = defaultdict(list)
        user_to_sl = defaultdict(list)
        # for (u, i), sl in self.ui_to_sid.items():
        for u, i in zip(self.train_uir[0].tolist(), self.train_uir[1].tolist()):
            if (u, i) not in self.ui_to_sid:
                continue
            sl = self.ui_to_sid[(u, i)]
            item_to_sl[i].extend(sl)
            user_to_sl[u].extend(sl)

        topic_to_sid = defaultdict(list)  # topic: u1, i1, u2, i2
        item_to_topic = defaultdict(list)
        user_to_topic = defaultdict(list)

        for i, sl in tqdm(item_to_sl.items(), desc='item-topic-sentence'):
            for sid in sl:
                if sid in self.sid_to_topic:
                    t = f'i{i}-t{self.sid_to_topic[sid]}'
                    topic_to_sid[t].append(sid)
                    item_to_topic[i].append(t)

        for u, sl in tqdm(user_to_sl.items(), desc='user-topic-sentence'):
            for sid in sl:
                if sid in self.sid_to_topic:
                    t = f'u{u}-t{self.sid_to_topic[sid]}'
                    topic_to_sid[t].append(sid)
                    user_to_topic[u].append(t)

        topic_to_id = {t: i+1 for i, t in enumerate(topic_to_sid.keys())}  # 将 user/item spcific topic 映射到ID
        print(f'Topic size: {len(topic_to_id):,}, where topics are specific to each users/items.')

        # gtid: global topic id, 为infomap聚类出来的 topic；与之相对的是 local topic id，为 topic + user/item 结合出来的
        id_to_gtid = [0] * (len(topic_to_id) + 1)
        for st, i in topic_to_id.items():
            id_to_gtid[i] = int(st.split('t')[1])

        id_to_gtid = torch.LongTensor(id_to_gtid)

        topic_to_sid = {topic_to_id[t]: s for t, s in topic_to_sid.items()}
        user_to_topic = {u: [topic_to_id[t] for t in ts] for u, ts in user_to_topic.items()}
        item_to_topic = {i: [topic_to_id[t] for t in ts] for i, ts in item_to_topic.items()}

        # _cast_fn = lambda ll: [torch.LongTensor(x) for x in zip(*ll.items())]

        def _cast_fn(x_to_list):
            r1, r2 = [], []
            for k, vs in tqdm(x_to_list.items()):
                for v in vs:
                    r1.append(k)
                    r2.append(v)
            return torch.LongTensor(r1), torch.LongTensor(r2)
        # import pdb; pdb.set_trace()

        # data_dict[('topic', 'topic_to_sentence', 'sentence')] = _cast_fn(topic_to_sid)
        # data_dict[('user', 'user_to_topic', 'topic')] = _cast_fn(user_to_topic)
        # data_dict[('item', 'item_to_topic', 'topic')] = _cast_fn(item_to_topic)

        data_dict[('sentence', 'sentence_to_topic', 'topic')] = _cast_fn(topic_to_sid)[::-1]
        data_dict[('topic', 'topic_to_user', 'user')] = _cast_fn(user_to_topic)[::-1]
        data_dict[('topic', 'topic_to_item', 'item')] = _cast_fn(item_to_topic)[::-1]

        # graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        graph = dgl.heterograph(data_dict)

        graph.nodes['topic'].data.update({'global_topic_id': id_to_gtid})

        # assign topic to each sentence
        sent_tid = [0] * graph.num_nodes('sentence')  # 只包含训练集的sentence

        for sid, tid in self.sid_to_topic.items():
            if sid >= len(sent_tid):
                continue
            sent_tid[sid] = tid
        graph.nodes['sentence'].data.update({'global_topic_id': torch.LongTensor(sent_tid)})

        make_labels = lambda x: torch.from_numpy(np.searchsorted(self.possible_rating_values, x).astype(np.int64))

        graph['train'].edata['rating'] = self.train_uir[2]
        graph['train'].edata['label'] = make_labels(self.train_uir[2])
        graph['train'].edata['review_id'] = train_rid_list
        graph['train'].edata['sentence_id'] = train_sid_list
        graph['valid'].edata['rating'] = self.valid_uir[2]
        graph['valid'].edata['label'] = make_labels(self.valid_uir[2])
        graph['valid'].edata['review_id'] = valid_rid_list
        graph['valid'].edata['sentence_id'] = valid_sid_list
        graph['test'].edata['rating'] = self.test_uir[2]
        graph['test'].edata['label'] = make_labels(self.test_uir[2])
        graph['test'].edata['review_id'] = test_rid_list
        graph['test'].edata['sentence_id'] = test_sid_list

        for rating in self.possible_rating_values:
            graph[str(rating)].edata['review_id'] = rating_to_rid[str(rating)]
            graph['rev-%s' % str(rating)].edata['review_id'] = rating_to_rid[str(rating)]

        def _calc_norm(x):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.sqrt(x))
            return x.unsqueeze(1)

        user_ci = []
        user_cj = []
        item_ci = []
        item_cj = []
        for r in self.possible_rating_values:
            r = str(r)
            user_ci.append(graph['rev-%s' % r].in_degrees())
            item_ci.append(graph[r].in_degrees())
            user_cj.append(graph[r].out_degrees())
            item_cj.append(graph['rev-%s' % r].out_degrees())
        user_ci = _calc_norm(sum(user_ci))
        item_ci = _calc_norm(sum(item_ci))
        user_cj = _calc_norm(sum(user_cj))
        item_cj = _calc_norm(sum(item_cj))
        graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
        graph.nodes['item'].data.update({'ci': item_ci, 'cj': item_cj})

        # asign candidate sentences to items
        for u, i in zip(self.valid_uir[0].tolist() + self.test_uir[0].tolist(), \
                        self.valid_uir[1].tolist() + self.test_uir[1].tolist()):
            if (u, i) not in self.ui_to_sid:
                continue
            sl = self.ui_to_sid[(u, i)]
            item_to_sl[i].extend(sl)

        item_max_sent_count = min_count(item_to_sl.values())
        print(f'item_max_sent_count: {item_max_sent_count}')
        item_candidate_sl = [_pad(item_to_sl[i], item_max_sent_count) for i in range(len(item_to_sl))]
        item_candidate_sl = torch.LongTensor(item_candidate_sl)
        graph.nodes['item'].data.update({'candidate_sentence_id': item_candidate_sl})
        return graph

    def get_whole_graphs(self):
        """ return encoder graph, train, valid and test graphs"""
        # etypes = [str(r) for r in self.possible_rating_values] + [f'rev-{r}' for r in self.possible_rating_values]
        # encoder_graph = dgl.edge_subgraph(self.graph, {x: torch.arange(self.graph.num_edges(x)) for x in etypes})
        encoder_graph = self.graph

        # train_decoder_graph = dgl.edge_subgraph(self.graph, {x: torch.arange(self.graph.num_edges(x)) for x in ['train']})['train']
        # valid_decoder_graph = dgl.edge_subgraph(self.graph, {x: torch.arange(self.graph.num_edges(x)) for x in ['valid']})['valid']
        # test_decoder_graph = dgl.edge_subgraph(self.graph, {x: torch.arange(self.graph.num_edges(x)) for x in ['test']})['test']

        train_decoder_graph = self.graph['train']
        valid_decoder_graph = self.graph['valid']
        test_decoder_graph = self.graph['test'] 

        return encoder_graph, train_decoder_graph, valid_decoder_graph, test_decoder_graph

    def create_a_dataloader(self, batch_size, num_layers, sample_etype, **kwargs):
        excluded_etypes = [('user', x, 'item') for x in ['train', 'valid', 'test']] \
                          + [('sentence', 'sentence_to_topic', 'topic'), \
                             ('topic', 'topic_to_user', 'user'), \
                             ('topic', 'topic_to_item', 'item')]
        excluder = EdgeTypeExcluder(self.graph, excluded_etypes)
        sampler = dgl.dataloading.NeighborSampler([-1]*num_layers)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude=excluder)
        eid_dict = {sample_etype: torch.arange(self.graph.number_of_edges(etype=sample_etype)) }
        dataloader = dgl.dataloading.DataLoader(self.graph, eid_dict, 
                                                graph_sampler=sampler, batch_size=batch_size, **kwargs)
        return dataloader

    def get_dataloaders(self, batch_size, num_layers):
        """ return train, valid and test dataloaders """
        return self.create_a_dataloader(batch_size, num_layers, 'train', shuffle=True, drop_last=True, num_workers=4), \
            self.create_a_dataloader(batch_size, num_layers, 'valid', shuffle=False, drop_last=False, num_workers=4), \
            self.create_a_dataloader(batch_size, num_layers, 'test', shuffle=False, drop_last=False, num_workers=4),

    def get_topic_sentence_sampler(self):
        fanouts = [{k: 0 for k in self.graph.canonical_etypes}] * 2
        fanouts[0][('sentence', 'sentence_to_topic', 'topic')] = -1
        fanouts[0][('topic', 'topic_to_user', 'user')] = -1
        fanouts[0][('topic', 'topic_to_item', 'item')] = -1
        sampler = dgl.dataloading.NeighborSampler(fanouts=fanouts)
        # sampler.sample(graph, {'user': torch.arange(5), 'item': torch.arange(5)})
        return sampler


class EdgeTypeExcluder:
    
    def __init__(self, g, exclude_edge_types):
        self.edge_types = {t: torch.arange(g.num_edges(t)) for t in exclude_edge_types}
    
    def __call__(self, x):
        return self.edge_types


if __name__ == '__main__':
    dataset_name = 'Clothing_5'
    dataset_path = '/root/RatingTopicGraph/Clothing_5/Clothing_5.json'
#    dataset_name = 'CDs_and_Vinyl_5'
#    dataset_path = '/home/d1/shuaijie/data/CDs_and_Vinyl_5/CDs_and_Vinyl_5.json'
    dataset = GraphData(dataset_name, dataset_path)

    g = dataset.graph
    # print(dataset.possible_rating_values)
    print(g)
    # encoder_graph, train_decoder_graph, valid_decoder_graph, test_decoder_graph = dataset.get_whole_graphs()
    # gs = dataset.get_dataloaders(1024, 2)
    # x = next(iter(gs[0]))
    # print('0', x[0])
    # print('1', x[1])
    # print('2', x[2])
#    import pdb; pdb.set_trace() 

