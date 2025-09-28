from commgnas.datasets.CommData import DATA
from commgnas.model.stack_gcn import StackGcn
import numpy as np
import torch
import random

graph = DATA()  #['gcn', 'sum', '8', '128', 'leaky_relu', 'gat', 'max', '6', '16', 'relu6']
# #data_set_name = ["cora", "citeseer", "wiki"]  #['gcn', 'sum', '6', '8', 'elu', 'cos', 'min', '6', '128', 'softplus']
# #gnn_list = [['const', 'sum', '1', '256', 'relu6', 'gat_sym', 'mean', '16', '64', 'elu'],   #["const", "mean", 4, 256, "relu6", "linear", "sum", 4, 64, "tanh"]
#             ['gat_sym', 'max', '16', '128', 'tanh', 'gat', 'max', '16', '64', 'elu'],
#             ["gcn", "mean", 2, 256, "leaky_relu", "gat", "sum", 2, 128, "sigmoid"]]
# #random_seed = [2, 46, 79]

data_set_name ="cora"
gnn_list = ['const', 'sum', '4', '256', 'relu6', 'linear', 'mean', '6', '8', 'linear']
random_seed =  2
def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deteriministic =True

#for data_name, gnn, seed in zip(data_set_name, gnn_list, random_seed):
set_seed(random_seed)
graph.get_data(data_set_name, shuffle_flag=False)
print("Graph Name:", data_set_name)
print("Gnn Architecture:", gnn_list)
model = StackGcn(graph,
                 gnn_architecture=gnn_list,
                 downstream_task_type="node_self_expressive_learning",
                 downstream_task_parameter={"lr": 0.001,
                                            "weight_decay": 0.00001,
                                            "se_epochs": 80,
                                            "se_loss_reg": 0.5,
                                            "patience": 40},
                 supervised_learning=False,
                 train_batch_size=50,
                 val_batch_size=10,
                 test_batch_size=10,
                 train_epoch=200,
                 train_epoch_test=200,
                 gnn_drop_out=0.6,
                 opt_parameter_dict={"learning_rate": 0.005, "l2_regularization_strength": 0.0005},
                 loss_type="self_supervised_loss",
                 val_evaluator_type="accuracy",
                 test_evaluator_type=["unsupervised_evaluate"])

model.evaluate()
print("\n\n")