import os
import configparser
from commgnas.auto_model import AutoModel
from commgnas.parallel import ParallelConfig
from commgnas.datasets.CommData import DATA

ParallelConfig(True)

graph = DATA()
# ["cora", "citeseer", "PubMed", "wiki",'computers','photo','coauthorcs','coauthorphy']
graph.get_data("pubmed", shuffle_flag=False)
config = configparser.ConfigParser()
config_path = os.path.dirname(os.path.abspath(__file__)) + \
              "/config/community_detection_config/graphpastune.ini"
config.read(config_path)
search_parameter = dict(config.items('search_parameter'))
gnn_parameter = dict(config.items("gnn_parameter"))
AutoModel(graph, search_parameter, gnn_parameter)

