import torch
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
import scipy as sp
class DownstreamTask(torch.nn.Module):

    def __init__(self,
                 downstream_task_parameter,
                 gnn_embedding_dim,
                 graph_data):
        super(DownstreamTask, self).__init__()
        self.graph_data = graph_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dim = graph_data.num_labels
        self.mlp = torch.nn.Linear(gnn_embedding_dim, output_dim)

    def forward(self,
                node_embedding_matrix,
                batch_x_index,
                mode="train"):

        cluster = KMeans(n_clusters=self.graph_data.num_labels)
        #cluster.fit(node_embedding_matrix.detach().cpu().numpy())
        y_pre = cluster.fit_predict(node_embedding_matrix.detach().cpu().numpy()) + 1
        y_pre = self.best_map(self.graph_data.test_y.detach().cpu().numpy(), y_pre)
        acc, nmi, f1 = self.err_rate(self.graph_data.test_y.detach().cpu().numpy(), y_pre)

        x = self.graph_data.test_x
        edge_index = self.graph_data.test_edge_index
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), edge_index.cpu().numpy()),
                                          shape=(num_nodes, num_nodes))  # 将边索引转换为稀疏矩阵,能够快速计算每个节点的度。
        torch_sparse_adj = torch.sparse_coo_tensor(edge_index, torch.ones(num_edges).to(self.device),
                                                   size=(num_nodes, num_nodes))  # 将边索引转换为稀疏张量
        degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float()  # 计算每个节点的度

        num_edges = int((edge_index.shape[1]) / 2)  # 计算边的数量
        Q = self.compute_fast_modularity(y_pre, num_nodes, num_edges, torch_sparse_adj, degree, self.device)
        print(" Acc:", acc," Q:",Q, "NMI:", nmi, "F1:", f1)

        # if mode == "train":
        #     predict_y = predict_y[self.graph_data.train_mask]
        # elif mode == "val":
        #     predict_y = predict_y[self.graph_data.val_mask]
        # elif mode == "test":
        #     predict_y = predict_y[self.graph_data.test_mask]
        # else:
        #     print("wrong mode")
        #     raise
        return acc, nmi, f1, Q

    def compute_fast_modularity(self, clusters, num_nodes, num_edges, torch_sparse_adj, degree,device):
        clusters = clusters.astype(int)
        mx = max(clusters)
        MM = np.zeros((num_nodes, mx+1 ))
        for i in range(len(clusters)):
            MM[i][clusters[i]] = 1
        MM = torch.tensor(MM).double().to(device)

        x = torch.matmul(torch.t(MM), torch_sparse_adj.double())
        x = torch.matmul(x, MM)
        x = torch.trace(x)

        y = torch.matmul(torch.t(MM), degree.double().to(device))
        y = torch.matmul(torch.t(y.unsqueeze(dim=0)), y.unsqueeze(dim=0))
        y = torch.trace(y)
        y = y / (2 * num_edges)
        return ((x - y) / (2 * num_edges)).item()

    def err_rate(self, y_true, y_pre):

        #y_pred = self.best_map(gt_s, s)
        acc = metrics.accuracy_score(y_true, y_pre)
        nmi = metrics.normalized_mutual_info_score(y_true, y_pre)
        f1_macro = metrics.f1_score(y_true, y_pre, average='macro')

        return [acc, nmi, f1_macro]

    def best_map(self, L1, L2):
        # L1 should be the groundtruth labels and L2 should be the clustering labels we got
        Label1 = np.unique(L1)
        nClass1 = len(Label1)
        Label2 = np.unique(L2)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = L1 == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = L2 == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        newL2 = np.zeros(L2.shape)
        for i in range(nClass2):
            newL2[L2 == Label2[i]] = Label1[c[i]]
        return newL2