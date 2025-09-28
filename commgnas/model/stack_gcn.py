import os

import torch
import warnings

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

warnings.filterwarnings("ignore")
from commgnas.model.stack_gcn_encoder.gcn_encoder import GcnEncoder
from commgnas.model.logger import gnn_architecture_performance_save_mul,\
                                  gnn_architecture_performance_save_sum,\
                                gnn_architecture_performance_save_noexpress,\
                                  test_performance_save
from commgnas.dynamic_configuration import optimizer_getter,  \
                                           loss_getter, \
                                           evaluator_getter, \
                                           downstream_task_model_getter
from torch_geometric.utils import dropout_adj, dropout_edge
from sklearn.preprocessing import normalize
import scipy as sp
from sklearn.manifold import TSNE
import numpy as np
import random
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.loader import NeighborSampler

class StackGcn(object):

    def __init__(self,
                 graph_data,
                 downstream_task_type="node_classification",
                 downstream_task_parameter={},
                 supervised_learning=True,
                 train_batch_size=128,
                 val_batch_size=1,
                 test_batch_size=1,
                 gnn_architecture=['gcn', 'sum',  1, 128, 'relu', 'gcn', 'sum', 1, 64, 'linear'],
                 gnn_drop_out=0.6,
                 train_epoch=100,
                 train_epoch_test=100,
                 bias=True,
                 early_stop=True,
                 early_stop_patience=5,
                 opt_type="adam",
                 opt_parameter_dict={"learning_rate": 0.005, "l2_regularization_strength": 0.0005},
                 loss_type="nll_loss",
                 val_evaluator_type="accuracy",
                 test_evaluator_type=["accuracy", "precision", "recall", "f1_value"]):

        self.graph_data = graph_data
        self.downstream_task_type = downstream_task_type
        self.downstream_task_parameter = downstream_task_parameter
        self.supervised_learning = supervised_learning
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.gnn_architecture = gnn_architecture
        self.gnn_drop_out = gnn_drop_out
        self.train_epoch = train_epoch
        self.train_epoch_test = train_epoch_test
        self.bias = bias
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.opt_type = opt_type
        self.opt_parameter_dict = opt_parameter_dict
        self.loss_type = loss_type
        self.val_evaluator_type = val_evaluator_type
        self.test_evaluator_type = test_evaluator_type
        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0
        self.min_loss=7
        self.best_idx=1
        self.stop_cnt=0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gnn_model = GcnEncoder(self.gnn_architecture,
                                    self.graph_data.num_features,
                                    dropout=self.gnn_drop_out,
                                    bias=self.bias).to(self.device)

        self.optimizer = optimizer_getter(self.opt_type,
                                          self.gnn_model,
                                          self.opt_parameter_dict)

        self.loss = loss_getter(self.loss_type)

        self.val_evaluator = evaluator_getter(self.val_evaluator_type)

        self.downstream_task_model = downstream_task_model_getter(self.downstream_task_type,
                                                                  self.downstream_task_parameter,
                                                                  int(self.gnn_architecture[-2]),
                                                                  self.graph_data).to(self.device)

    def fit(self, mode='train'):

        x = self.graph_data.test_x
        edge_index = self.graph_data.test_edge_index
        drop_edge_rate1 = 0.2
        drop_edge_rate2 = 0.4
        x1 = x 
        x2 = x
        edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate1)[0]
        edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate2)[0]
        x1_node_nums = x1.shape[0]
        edge_index_2[0] = edge_index_2[0] + x1_node_nums
        edge_index_2[1] = edge_index_2[1] + x1_node_nums
        train_x = torch.cat([x1, x2], dim=0)
        train_edge_index = torch.cat([edge_index_1, edge_index_2], dim=-1)

        sup_loss = 0.0
        mod_loss = 0.0
        train_loss_value = 0.0

        first_loss = 0.0
        first_mod_loss = 0.0
        first_sup_loss = 0.0

        final_loss = 0.0
        final_mod_loss = 0.0
        final_sup_loss = 0.0

        best_loss = 1e9
        bad_count = 0

        for epoch in range(1, self.train_epoch+1 ):
            #小数据集
            # node_embedding_matrix = self.gnn_model(train_x, train_edge_index) #大图节点数为2*N
            # node_embedding_matrix_ = node_embedding_matrix.view(2, self.graph_data.node_nums, -1)
            # z1 = node_embedding_matrix_[0]
            # z2 = node_embedding_matrix_[1]
            # sup_loss = self.loss.function(z1, z2)
            # mod_loss_1 = self.modularity_loss(z1, edge_index)
            # mod_loss_2 = self.modularity_loss(z2, edge_index)
            # #综合模块度
            # mod_loss = (mod_loss_1 + mod_loss_2)/2
            # train_loss = sup_loss +10*mod_loss #train_loss = sup_loss
            #大数据集  Pubmed
            # 1. 采样节点 (使用原始图的节点数)
            batch_size = 1000  # 根据显存调整
            num_nodes = self.graph_data.test_x.size(0)
            node_idx = torch.randperm(num_nodes)[:batch_size]
            
            # 2. 创建子图
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[node_idx] = True
            
            node_mask = node_mask.to(self.device)

            # 3. 创建两个视图的增强子图
            # 视图1
            edge_index1, _ = dropout_edge(edge_index, p=drop_edge_rate1)
            edge_mask1 = node_mask[edge_index1[0]] & node_mask[edge_index1[1]]
            sub_edge_index1 = edge_index1[:, edge_mask1]
            
            # 视图2
            edge_index2, _ = dropout_edge(edge_index, p=drop_edge_rate2)
            edge_mask2 = node_mask[edge_index2[0]] & node_mask[edge_index2[1]]
            sub_edge_index2 = edge_index2[:, edge_mask2]
            
            # 4. 节点特征索引
            sub_x = x[node_idx]
            
            # 5. 重新映射节点索引 (两个视图使用相同的节点)
            mapping = torch.zeros(num_nodes, dtype=torch.long)
            mapping[node_idx] = torch.arange(len(node_idx))
            mapping = mapping.to(self.device)
            sub_edge_index1 = mapping[sub_edge_index1]
            sub_edge_index2 = mapping[sub_edge_index2]
            
            # 6. 前向传播 - 两个视图分别处理
            # 视图1
            emb1 = self.gnn_model(sub_x, sub_edge_index1)
            # 视图2
            emb2 = self.gnn_model(sub_x, sub_edge_index2)
            
            # 7. 计算损失
            sup_loss = self.loss.function(emb1, emb2)  # 对比损失
            
            # 模块度损失 (使用子图的边)
            mod_loss_1 = self.modularity_loss(emb1, sub_edge_index1)
            mod_loss_2 = self.modularity_loss(emb2, sub_edge_index2)
            mod_loss = (mod_loss_1 + mod_loss_2) / 2
            
            train_loss = sup_loss + 10 * mod_loss



            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            if mode == "test":
                if train_loss.item() < best_loss:
                    bad_count = 0
                    best_loss = train_loss.item()
                else:
                    bad_count += 1
                    if bad_count == self.early_stop_patience:
                        break
            train_loss_value = train_loss.item()
            if epoch == 1:
                first_loss = train_loss_value
                first_mod_loss = mod_loss.item()
                first_sup_loss = sup_loss.item()
                print(
                    f"epoch:{epoch}, train loss:{train_loss_value}, supervised loss:{first_sup_loss}, modularity loss:{first_mod_loss}")
            if epoch % 5 == 0:
                print(f"epoch:{epoch}, train loss:{train_loss_value}, supervised loss:{sup_loss.item()}, modularity loss:{mod_loss.item()}")
        if mode == "test":
            # with torch.no_grad():
            #     final_embeddings = self.gnn_model(self.graph_data.test_x, self.graph_data.test_edge_index)
            # return final_embeddings
            pass
        else:
            final_loss = train_loss_value
            # final_mod_loss = mod_loss.item()
            # final_sup_loss = sup_loss.item()
            
            print("6666666")
            feedback = final_loss
            print("feedback:", feedback)
            gnn_architecture_performance_save_noexpress(self.gnn_architecture, feedback, self.graph_data.data_name)
            #return feedback
            return {"feedback": feedback, "L_1": first_loss,  "M_1": first_mod_loss, "S_1": first_sup_loss,"L_2": final_loss, "M_2": final_mod_loss, "S_2": final_sup_loss}

    def evaluate(self):
        self.train_epoch = self.train_epoch_test

        self.fit(mode="test")

        self.downstream_task_model = downstream_task_model_getter("node_cluster",
                                                                  self.downstream_task_parameter,
                                                                  int(self.gnn_architecture[-2]),
                                                                  self.graph_data).to(self.device)

        self.gnn_model.eval()
        #小图
        #Z = self.gnn_model(self.graph_data.test_x, self.graph_data.test_edge_index)   
        #大图
        all_embeddings = []
        batch_size = 2000  # 评估批次大小（可调整）
        num_nodes = self.graph_data.test_x.size(0)
        # 分批处理所有测试节点
        for i in range(0, num_nodes, batch_size):
            # 获取当前批次的节点索引
            node_idx = torch.arange(i, min(i + batch_size, num_nodes))
            
            # 获取当前批次的子图
            with torch.no_grad():
                # 1. 获取当前批次的节点特征
                batch_x = self.graph_data.test_x[node_idx]
                
                # 2. 创建包含当前批次及其邻居的子图
                # 使用k-hop邻居采样确保信息完整
                batch_edge_index, _, edge_mask = k_hop_subgraph(
                    node_idx, 
                    num_hops=2,  # 根据GNN层数设置（层数+1）
                    edge_index=self.graph_data.test_edge_index,
                    relabel_nodes=True
                )
                
                # 3. 映射回原始特征索引
                # 获取子图中的所有节点（包括邻居）
                subgraph_nodes = batch_edge_index.unique()
                sub_x = self.graph_data.test_x[subgraph_nodes]
                
                # 4. 在子图上进行推理
                z_sub = self.gnn_model(sub_x, batch_edge_index)
                
                # 5. 提取目标节点的嵌入
                # 在子图中，目标节点位于索引0到len(node_idx)-1
                batch_embeddings = z_sub[:len(node_idx)]
                all_embeddings.append(batch_embeddings.cpu())
        
        # 拼接所有批次的嵌入
        Z = torch.cat(all_embeddings, dim=0)












        











        #acc, nmi, f1, y_pred_label, S ,Q = self.downstream_task_model(Z,None,mode="test")
        acc, nmi, f1,   Q, s, label ,z= self.downstream_task_model(Z, None, mode="test")

        test_performance_dict = {"accuracy": acc,
                                 "modurality":Q,
                                 "NMI": nmi,
                                 "F1": f1}
        hyperparameter_dict = self.downstream_task_parameter
        test_performance_save(self.gnn_architecture,
                              test_performance_dict,
                              hyperparameter_dict,
                              self.graph_data.data_name)
        matplotlib.use('Agg')  # 切换到无头模式
        # custom_colors = ['#E89DA0', '#88CEE6', '#F6C8AB', '#B2D3A4', '#80C1C4', '#B696B6','#E6CECF']  # 示例颜色，按需修改
        # custom_cmap = ListedColormap(custom_colors)
        #tsne = TSNE(n_components=2, metric='precomputed', perplexity=5, init="random", random_state=42)
        tsne = TSNE(n_components=2, init="random", random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(z.cpu().detach().numpy())
        # 绘制聚类图
        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=label,  cmap='viridis', s=40, alpha=0.8)
        plt.colorbar(scatter, label='Cluster Label')
        plt.title("Cora Clustering Visualization ")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.savefig("citeseer_0.5.png")  # 保存图片到文件
        plt.show()
    def modularity_loss(self, output, edge_index):
        num_nodes = output.size(0)
        num_edges_ = edge_index.shape[1]
        sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges_), edge_index.cpu().numpy()),
                                          shape=(num_nodes, num_nodes))  # 将边索引转换为稀疏矩阵
        degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float().to(device)  # 计算每个节点的度
        num_edges = int((edge_index.shape[1]) / 2)
        # 随机采样
        sample_size = int(1 * num_nodes)
        s = random.sample(range(0, num_nodes), sample_size)
        s_output = output[s, :]  # 采样节点的输出
        s_adj = sparse_adj[s, :][:, s]  # 采样节点的邻接矩阵
        s_adj = self.convert_scipy_torch_sp(s_adj)  # 将邻接矩阵转换为稀疏张量
        s_degree = degree[s]  # 采样节点的度

        x = torch.matmul(torch.t(s_output).double(), s_adj.double().to(device))  # 经过邻接关系加权后的节点特征矩阵
        x = torch.matmul(x, s_output.double())  # 计算x
        x = torch.trace(x)  # 计算x的迹


        y = torch.matmul(torch.t(s_output).double(), s_degree.double().to(device))
        y = (y ** 2).sum()
        y = y / (2 * num_edges)
        scaling = num_nodes ** 2 / (sample_size ** 2)
        loss = -((x - y) / (2 * num_edges)) * scaling  # 计算模块度损失
        return loss
    def convert_scipy_torch_sp(self,sp_adj):
        sp_adj = sp_adj.tocoo()
        indices = torch.tensor(np.vstack((sp_adj.row, sp_adj.col)))
        sp_adj = torch.sparse_coo_tensor(indices, torch.tensor(sp_adj.data), size=sp_adj.shape)
        return sp_adj




if __name__=="__main__":
   pass