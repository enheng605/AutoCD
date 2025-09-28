import torch
import numpy as np
from munkres import Munkres
from sklearn import metrics
from sklearn import cluster
from scipy.sparse.linalg import svds
from torch.nn.parameter import Parameter
from sklearn.preprocessing import normalize
import scipy as sp
class DownstreamTask(torch.nn.Module):

    def __init__(self,
                 downstream_task_parameter,
                 gnn_embedding_dim,
                 graph_data):

        super(DownstreamTask, self).__init__()
        self.data = graph_data
        self.downstream_task_parameter = downstream_task_parameter
        #self.clusters=self.downstream_task_parameter["num_clusters"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.semodel = SelfExpr(self.data.test_x.shape[0]).to(self.device)
        self.seoptimizer = torch.optim.Adam(self.semodel.parameters(),
                                            lr=self.downstream_task_parameter["lr"],
                                            weight_decay=self.downstream_task_parameter['weight_decay'])

    def forward(self,
                node_embedding_matrix,
                batch_x_index,
                mode="train"):

        Z = torch.tensor(normalize(node_embedding_matrix.cpu().detach().numpy())).to(self.device)

        print("\n Starting self representation training !")
        best_accuracy, best_nmi, best_f1,best_q,best_s,best_label,best_z= self.self_representation(Z, self.data.num_labels)
        print("  Acc:", best_accuracy, "  Q:", best_q, " NMI:", best_nmi, " F1:", best_f1)
        #S_ = torch.matmul(Z, Z.T)
        #S = torch.sigmoid(S_)
        # print("\n Performance estimation of spectral clustering on the similarity matrix !")
        # scores = self.spectral_cluster(S, self.data.test_y, self.data.num_labels)   #y_pred从0开始的索引
        #
        # accuracy = scores[0]
        # NMI = scores[1]
        # F1 = scores[2]
        # y_pred_label = scores[3]


        # x = self.data.test_x
        # edge_index = self.data.test_edge_index
        # num_nodes = x.shape[0]
        # num_edges = edge_index.shape[1]
        # sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), edge_index.cpu().numpy()),
        #                                   shape=(num_nodes, num_nodes))  # 将边索引转换为稀疏矩阵,能够快速计算每个节点的度。
        # torch_sparse_adj = torch.sparse_coo_tensor(edge_index, torch.ones(num_edges).to(self.device),
        #                                            size=(num_nodes, num_nodes))  # 将边索引转换为稀疏张量
        # degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float()  # 计算每个节点的度
        #
        # num_edges = int((edge_index.shape[1]) / 2)  # 计算边的数量
        # Q = self.compute_fast_modularity(y_pred_label, num_nodes, num_edges, torch_sparse_adj, degree,self.device)
        # print(" Acc:", scores[0]," Q:",Q, "NMI:", scores[1], "F1:", scores[2])



        return best_accuracy, best_nmi, best_f1,best_q,best_s,best_label,best_z
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

    def self_representation(self, Z, n_class):

        max_epoch = self.downstream_task_parameter['se_epochs']
        alpha = self.downstream_task_parameter['se_loss_reg']
        patience = self.downstream_task_parameter['patience']
        best_loss = 1e9
        bad_count = 0
        best_C = 0.0
        best_accuracy = -1.0
        best_nmi = -1.0
        best_f1 = -1.0
        best_q = -1.0
        best_s=None
        best_label = None
        best_z=None
        for epoch in range(max_epoch):
            self.semodel.train()
            self.seoptimizer.zero_grad()
            C, CZ = self.semodel(Z)
            se_loss = torch.norm(Z - CZ)
            reg_loss = torch.norm(C)
            loss = se_loss + alpha * reg_loss
            loss.backward()
            train_loss_value = loss.item()
            if epoch % 10 == 0:
                print("self representation learning train epoch: ", epoch, " train loss value: ", train_loss_value)
            self.seoptimizer.step()
            C = C.cpu().detach().numpy()
            S = self.similarity_matrix_computation(C, n_class, 4)  #计算相似矩阵
            scores = self.spectral_cluster(S, self.data.test_y, self.data.num_labels)  # y_pred从0开始的索引
            accuracy = scores[0]
            NMI = scores[1]
            F1 = scores[2]
            y_pred_label = scores[3]

            x = self.data.test_x
            edge_index = self.data.test_edge_index
            num_nodes = x.shape[0]
            num_edges = edge_index.shape[1]
            sparse_adj = sp.sparse.csr_matrix((np.ones(num_edges), edge_index.cpu().numpy()),
                                              shape=(num_nodes, num_nodes))  # 将边索引转换为稀疏矩阵,能够快速计算每个节点的度。
            torch_sparse_adj = torch.sparse_coo_tensor(edge_index, torch.ones(num_edges).to(self.device),
                                                       size=(num_nodes, num_nodes))  # 将边索引转换为稀疏张量
            degree = torch.tensor(sparse_adj.sum(axis=1)).squeeze().float()  # 计算每个节点的度

            num_edges = int((edge_index.shape[1]) / 2)  # 计算边的数量
            Q = self.compute_fast_modularity(y_pred_label, num_nodes, num_edges, torch_sparse_adj, degree, self.device)
            print(" Acc:", accuracy, " Q:", Q, "NMI:", NMI, "F1:", F1)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_s=S
                best_label = y_pred_label
                best_z=CZ
                # best_nmi = NMI
                # best_f1 = F1
                # best_q = Q
            if NMI > best_nmi:
                best_nmi = NMI
            if F1 > best_f1:
                best_f1 = F1
            if Q > best_q:
                best_q = Q
        return best_accuracy, best_nmi, best_f1,best_q,best_s,best_label,best_z

    def similarity_matrix_computation(self, C, K, d):

        # C: coefficient matrix,
        # K: number of clusters,
        # d: dimension of each subspace

        C_star = 0.5 * (C + C.T)  #构建对称矩阵
        r = min(d * K + 1, C_star.shape[0] - 1)  #计算特征分解的维度
        U, Sig, _ = svds(C_star, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        Sig_sqrt = np.sqrt(Sig[::-1])
        Sig_sqrt = np.diag(Sig_sqrt)
        R_star = U.dot(Sig_sqrt)
        R_star = normalize(R_star, norm='l2', axis=1)
        R_star = R_star.dot(R_star.T)
        R_star = R_star * (R_star > 0)
        S = np.abs(R_star)
        S = 0.5 * (S + S.T)
        S = S / S.max()

        return S

    def spectral_cluster(self, S, test_true_y, n_class):

        y_prediction = self.cluster(S, n_class)
        #print("Spectral clustering done.. finding fest match based on Kuhn-Munkres")
        scores = self.err_rate(test_true_y.detach().cpu().numpy(), y_prediction)
        return scores

    def err_rate(self, test_true_y, y_prediction):

        y_pred = self.best_match(test_true_y, y_prediction)
        acc = metrics.accuracy_score(test_true_y, y_pred)
        nmi = metrics.normalized_mutual_info_score(test_true_y, y_pred)
        f1_macro = metrics.f1_score(test_true_y, y_pred, average='macro')

        return [acc, nmi, f1_macro, y_pred]

    def cluster(self, S, K):

        # S: similarity matrix,
        # K: number of clusters,
        # d: dimension of each subspace

        spectral = cluster.SpectralClustering(n_clusters=K,   #谱聚类
                                              eigen_solver='arpack',
                                              affinity='precomputed',
                                              assign_labels='discretize')

        output = spectral.fit_predict(S) + 1   #变换成从1开始的索引
        return output

    def best_match(self, L1, L2):

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


class SelfExpr(torch.nn.Module):

    def __init__(self, n):
        self.n = n
        super(SelfExpr, self).__init__()
        self.C_ = Parameter(torch.FloatTensor(n, n).uniform_(0, 0.01))

    def forward(self, Z):

        C = self.C_ - torch.diag(torch.diagonal(self.C_))
        output = torch.mm(C, Z)

        return C, output