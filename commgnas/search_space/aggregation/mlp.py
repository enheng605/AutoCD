import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class Aggregation(nn.Module):
    """
    Realizing a simple MLP aggregation manner for the source_node_representation_with_coefficient
    using a single linear layer.
    """

    def __init__(self):
        super(Aggregation, self).__init__()

    def function(self, source_node_representation_with_coefficient, edge_index,feature_dim):
        # 首先进行均值聚合
        aggregated_representation = scatter_mean(source_node_representation_with_coefficient, edge_index[1], dim=0)
        # 通过线性层进行处理
        layer = nn.Linear(feature_dim, feature_dim)    #指定输入维度和输出维度    输入：data特征维度    输出：不好取
        layer.to('cuda:0')
        output_representation = layer(aggregated_representation)  # 应用线性层
        return output_representation
