import csv
import json
import os
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj
from torch_geometric.datasets import OGB_MAG
# from ogb.nodeproppred import PygNodePropPredDataset
# import kagglehub
# import pandas as pd
import os.path as osp
from torch_geometric.data import Data
def load_npz(filepath):
    filepath = osp.abspath(osp.expanduser(filepath))

    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    if osp.isfile(filepath):
        with np.load(filepath, allow_pickle=True) as loader:
            loader = dict(loader)
            for k, v in loader.items():
                if v.dtype.kind in {'O', 'U'}:
                    loader[k] = v.tolist()
            return loader
    else:
        raise ValueError(f"{filepath} doesn't exist.")
class DATA(object):

    def __init__(self):

        self.name = ["cora", "citeseer", "pubmed", "wiki",'computers','photo','coauthorcs','coauthorphy']

    def get_data(self,
                 dataset,
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=123):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_name = dataset

        if data_name in ["cora", "citeseer", "pubmed"]:
            path = os.path.split(os.path.realpath(__file__))[0]+'/CITE/'
            dataset = Planetoid(path, dataset)
            data = dataset[0]
        elif data_name in ["computers", "photo", "coauthorcs", "coauthorphy"]:
            path = os.path.split(os.path.realpath(__file__))[0] + '/'
            # dataset = Planetoid(path, "Pubmed")
            dataset = load_npz(path + data_name + '.npz')
            adj_matrix = torch.from_numpy(dataset['adj_matrix'].toarray())
            edge_index = torch.nonzero(adj_matrix).t()
            # 将稀疏矩阵转换为 PyTorch 稀疏张量
            node_attr_coo = dataset['node_attr'].tocoo()
            row = node_attr_coo.row
            col = node_attr_coo.col
            data = node_attr_coo.data
            x = torch.sparse.FloatTensor(
                torch.LongTensor([row, col]),  # 行和列索引
                torch.FloatTensor(data),  # 非零元素值
                torch.Size(dataset['node_attr'].shape)  # 稀疏矩阵的形状
            )
            x_dense = x.to_dense()
            y = torch.tensor(dataset['node_label'], dtype=torch.int)
            num_nodes = x.shape[0]
            num_features = x.shape[1]
            data = Data(x=x_dense, edge_index=edge_index, y=y, num_features=num_features)
            data.num_classes = torch.unique(data.y).size(0)

            num_train = int(0.8 * num_nodes)  # 80%的节点用于训练
            num_val = int(0.1 * num_nodes)  # 10%的节点用于验证
            indices = torch.randperm(num_nodes)
            train_indices = indices[:num_train]  # 训练集节点的索引
            val_indices = indices[num_train:num_train + num_val]  # 验证集节点的索引
            test_indices = indices[num_train + num_val:]  # 测试集节点的索引
            # 创建掩码
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)  # 初始为全False
            train_mask[train_indices] = True  # 训练集的节点设置为True

            val_mask = torch.zeros(num_nodes, dtype=torch.bool)  # 初始为全False
            val_mask[val_indices] = True  # 验证集的节点设置为True

            test_mask = torch.zeros(num_nodes, dtype=torch.bool)  # 初始为全False
            test_mask[test_indices] = True  # 测试集的节点设置为True

            # 将掩码添加到 data 中
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        else:
            data = self.load_wiki()

        edge_index = data.edge_index.to(device)
        x = data.x.to(device)
        y = data.y.to(device)

        index_list = [i for i in range(y.size(0))]

        # construct transductive node classification task mask
        if shuffle_flag:

            if not random_seed:
                random_seed = 123
            random.seed(random_seed)

            random.shuffle(index_list)

            if train_splits == None or val_splits == None:

                train_splits = self.count_(data.train_mask)
                val_splits = self.count_(data.val_mask)
                test_splits = self.count_(data.test_mask)

                idx_train = index_list[:train_splits]
                idx_val = index_list[train_splits:train_splits+val_splits]
                idx_test = index_list[train_splits+val_splits:train_splits+val_splits+test_splits]

            else:

                idx_train = index_list[:int(y.size(0) * train_splits)]
                idx_val = index_list[int(y.size(0) * train_splits):int(y.size(0) * train_splits) + int(y.size(0) * val_splits)]
                idx_test = index_list[int(y.size(0) * train_splits) + int(y.size(0) * val_splits):]
        else:

            if train_splits == None or val_splits == None:

                train_splits = self.count_(data.train_mask)
                val_splits = self.count_(data.val_mask)
                test_splits = self.count_(data.test_mask)

                idx_train = index_list[:train_splits]
                idx_val = index_list[train_splits:train_splits + val_splits]
                idx_test = index_list[train_splits + val_splits:train_splits + val_splits + test_splits]

            else:

                idx_train = index_list[:int(y.size(0) * train_splits)]
                idx_val = index_list[int(y.size(0) * train_splits):int(y.size(0) * train_splits) + int(y.size(0) * val_splits)]
                idx_test = index_list[int(y.size(0) * train_splits) + int(y.size(0) * val_splits):]

        self.train_mask = torch.tensor(self.sample_mask_(idx_train, y.size(0)), dtype=torch.bool)
        self.val_mask = torch.tensor(self.sample_mask_(idx_val, y.size(0)), dtype=torch.bool)
        self.test_mask = torch.tensor(self.sample_mask_(idx_test, y.size(0)), dtype=torch.bool)

        # construct random x1, x2, Edge1, Edge2

        drop_feature_rate1 = 0.3
        drop_feature_rate2 = 0.4
        drop_edge_rate1 = 0.2
        drop_edge_rate2 = 0.4

        x1 = self.drop_feature(x, drop_feature_rate1)
        x2 = self.drop_feature(x, drop_feature_rate2)

        edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate1)[0]
        edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate2)[0]

        # combine x1, x2 to a whole graph dataset
        x1_node_nums = x1.shape[0]

        edge_index_2[0] = edge_index_2[0] + x1_node_nums
        edge_index_2[1] = edge_index_2[1] + x1_node_nums

        # combine x1 and x2
        x_ = torch.cat([x1, x2], dim=0)

        # combine edge_index_1 and edge_index_2:
        edge_index_ = torch.cat([edge_index_1, edge_index_2], dim=-1)

        # Auto-GNAS input required attribution

        self.test_y = y

        self.train_x = x_
        self.val_x = x_

        self.test_x = x

        self.train_edge_index = edge_index_
        self.val_edge_index = edge_index_

        self.test_edge_index = edge_index

        self.num_features = data.num_features
        self.num_labels = y.max().item() + 1
       #self.num_labels=3
        self.data_name = data_name
        self.node_nums = x1_node_nums
        return data
    def sample_mask_(self, idx, l):
        """ create mask """
        mask = np.zeros(l)
        for index in idx:
            mask[index] = 1
        return np.array(mask, dtype=np.int32)

    def count_(self, mask):
        true_num = 0
        for i in mask:
            if i:
                true_num += 1
        return true_num

    def drop_feature(self, x, drop_prob):
        drop_mask = torch.empty(
            (x.size(1),),
            dtype=torch.float32,
            device=x.device).uniform_(0, 1) < drop_prob
        x = x.clone()
        x[:, drop_mask] = 0
        return x
    def load_dezzer_lastfm(self,data_name ):
        path = os.path.split(os.path.realpath(__file__))[0] + "/CITE/"
        dataset = Planetoid(path, "cora")
        data = dataset[0]
        #root_path = os.path.split(os.path.realpath(__file__))[0]
        if data_name == "deezer":
            # replace with actual data from deezer
            with open(path +'deezer/features.json', 'r') as f:
                features_data = json.load(f)
            max_dimension = 0
            for features in features_data.values():
                dimension = len(features)  # 计算当前节点的特征维度
                if dimension > max_dimension:  # 更新最大维度
                    max_dimension = dimension
            features = 0 * torch.FloatTensor(28281, max_dimension)
            adj = 0 * torch.LongTensor(2, 92752)
            labels = 0 * torch.LongTensor(28281)

        else:  #lastfm
            with open(path +'lastfm/features.json', 'r') as f:
                features_data = json.load(f)
            max_dimension = 0
            for features in features_data.values():
                dimension = len(features)  # 计算当前节点的特征维度
                if dimension > max_dimension:  # 更新最大维度
                    max_dimension = dimension
            features = 0 * torch.FloatTensor(7624, max_dimension)
            adj = 0 * torch.LongTensor(2, 27806)
            labels = 0 * torch.LongTensor(7624)

        with open(path +data_name+'/edges.csv', 'r') as f:
            reader = csv.reader(f)
            i = 0
            for row in reader:
                adj[0, i] = int(row[0])
                adj[1, i] = int(row[1])
                i += 1
        # with open(root_path + '/commgnas/datasets/CITE/'+self.name+'/features.json', 'r') as f:
        #     features_data = json.load(f)
        for i, (key, value) in enumerate(features_data.items()):
            if value is None or len(value) == 0:  # 如果值为空，填充零（已在初始化中完成）
                continue
            else: # 否则，将特征值放入矩阵中
                features[i, :len(value)] = torch.tensor(value, dtype=torch.float)
        with open(path +data_name+'/target.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node = int(row[0])
                label = int(row[1])
                # labels 12 and 14 are missing in data. Rename 18 and 19 to 12 and 14
                # if label == 18:
                #     label = 12
                # if label == 19:
                #     label = 14
                labels[node] = label

        data.x = features
        data.y = labels
        data.edge_index = adj
        data.num_features = features.shape[1]
        return data



    def load_wiki(self):
        # load a dummy dataset to return the data in the same format as
        # those available in pytorch geometric
        path = os.path.split(os.path.realpath(__file__))[0] + "/CITE/"
        dataset = Planetoid(path, "cora")
        data = dataset[0]

        #wiki_path = os.path.split(os.path.realpath(__file__))[0]

        # replace with actual data from Wiki
        features = 0 * torch.FloatTensor(2405, 4973)
        adj = 0 * torch.LongTensor(2, 17981)
        labels = 0 * torch.LongTensor(2405)

        with open(path +'wiki/graph.txt', 'r') as f:
            i = 0
            for line in f:
                temp_list = line.split()
                adj[0, i] = int(temp_list[0])
                adj[1, i] = int(temp_list[1])
                i += 1

        with open(path +'wiki/tfidf.txt', 'r') as f:
            i = 0
            for line in f:
                temp_list = line.split()
                u = int(temp_list[0])
                v = int(temp_list[1])
                features[u, v] = float(temp_list[2])
                i += 1

        with open(path +'wiki/group.txt', 'r') as f:
            i = 0
            for line in f:
                temp_list = line.split()
                node = int(temp_list[0])
                label = int(temp_list[1])
                # labels 12 and 14 are missing in data. Rename 18 and 19 to 12 and 14
                if label == 18:
                    label = 12
                if label == 19:
                    label = 14
                labels[node] = label - 1
                i += 1

        data.x = features
        data.y = labels
        data.edge_index = adj
        data.num_features = features.shape[1]
        return data

if __name__=="__main__":

    graph = DATA()
    graph.get_data("lastfm", shuffle_flag=False)
    pass