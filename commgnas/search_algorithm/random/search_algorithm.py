import json
import os
import random

from commgnas.parallel import ParallelOperater
list = [
            {0: 'gat', 1: 'gat_sym', 2: 'gcn', 3: 'const', 4: 'generalized_linear', 5: 'cos', 6: 'linear'},
            {0: 'sum', 1: 'mean', 2: 'max', 3: 'mlp', 4: 'mean', 5: 'max', 6: 'sum'},
            {0: '1', 1: '2', 2: '4', 3: '4', 4: '6', 5: '8', 6: '16'},
            {0: '8', 1: '16', 2: '32', 3: '64', 4: '128', 5: '256', 6: '64'},
            {0: 'sigmoid', 1: 'tanh', 2: 'relu6', 3: 'leaky_relu', 4: 'softplus', 5: 'elu', 6: 'linear'}
        ]
class Search(object):
    def __init__(self, data, search_parameter, gnn_parameter, search_space):
        self.num_samples=10
        self. num_layers=2
        self.dataname = search_parameter['dataname']
        self.parallel_estimation = ParallelOperater(data, gnn_parameter)

    def generate_gnn_architecture(self):
        num_operators = len(list)
        arch_list = [
            [random.randint(0, len(list[i % num_operators]) - 1) for i in range(self.num_layers * num_operators)]
            for _ in range(self.num_samples)
        ]
        return arch_list

    def map_indices_to_strings(self,arch_list, list2):
        mapped_arch_list = []
        for arch in arch_list:
            mapped_arch = []
            for i, index in enumerate(arch):
                layer_index = i % len(list)  # Determine which operator set to use
                mapped_arch.append(list[layer_index][index])  # Map index to string
            mapped_arch_list.append(mapped_arch)
        return mapped_arch_list

    def search_operator(self):
        arch_list=[]
        feedback_list = []
        performance_history = []
        root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        filename = '/history/gpt/' + self.dataname + '/'
        filename_performance = root_directory +  filename + "random.json"
        arch_list = self.generate_gnn_architecture()
        string_arch_list = self.map_indices_to_strings(arch_list, list)
        result = self.parallel_estimation.estimation(string_arch_list)
        feedback_list.append(result)
        feedback_values = feedback_list[0]
        for arch, feedback in zip(arch_list, feedback_values):
            performance_history.append({"arch": arch, "feedback": round(feedback, 4)})
        with open(filename_performance, "w") as f:  # 会自动创建
            json.dump(performance_history, f)