import os
path = os.path.split(os.path.realpath(__file__))[0]
root_directory =os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(path)
print(root_directory)


# #### note:
#         Once again, your task is to help me find the optimal model on the dataset of ''' + self.dataname + '''. The main difficulty of this task lies in how to reasonably arrange the search strategies in different stages. We should choose a new model to try based on the existing model and its corresponding feedback, so as to iteratively find the best model.
#         At the beginning, when there were few experimental results, we in the Exploration phase, we need to explore the search space and identify which model is promising. We can #randomly# select a batch of models and evaluate their performance. Afterwards, we can sort the models based on their feedback and select some well performing models as candidates for our Exploitation phase.
#         When we have a certain amount of experimental results, we are in the Exploitation phase, we focus on improving search by exploring the search space more effectively. We can use optimization algorithms, such as Bayesian optimization or evolutionary algorithms, to focus on those models that perform best among the models that have been searched for experimental results, rather than randomly selecting them.

# The definition of search space for models follow the neighborhood aggregation schema, i.e., the Message Passing Neural Network (MPNN), which is formulated as:
#         {
#             $$\mathbf{m}_{v}^{k+1}=AGG_{k}(\{M_{k}(\mathbf{h}_{v}^{k}\mathbf{h}_{u}^{k},\mathbf{e}_{vu}^{k}):u\in N(v)\})$$
#             $$\mathbf{h}_{v}^{k+1}=ACT_{k}(COM_{k}(\{\mathbf{h}_{v}^{k},\mathbf{m}_{v}^{k+1}\}))$$
#         }
#         where $k$ denotes $k$-th layer, $N(v)$ denotes a set of neighboring nodes of $v$, $\mathbf{h}_{v}^{k}$, $\mathbf{h}_{u}^{k}$ denotes hidden embeddings for $v$ and $u$ respectively, $\mathrm{e}_{vu}^{k}$ denotes features for edge e(v, u) (optional), $\mathbf{m}_{v}^{k+1}$denotes the intermediate embeddings gathered from neighborhood $N(v)$, $M_k$ denotes the message function, $AGG_{k}$ denotes the neighborhood aggregation function, $COM_{k}$ denotes the combination function between intermediate embeddings and embeddings of node $v$ itself from the last layer, $ACT_{k}$ denotes activation function. Such message passing phase in repeats for $L$ times (i.e.,$ k\in\{1,\cdots,L\}$).
#You should give 10 different models at a time, one model contains #10# operations.
#For convenience, the candidate value corresponding to each component is represented by a number, ranging from #0 to 6#

# The components that need to be determined in the search space through architecture search, along with their corresponding candidate values are as follows：
#         {
#             attention: [gat,gat_sym,gcn,const,generalized_linear,cos,linear];
#             aggregation: [sum,mean,max,mlp,mean,max,sum];
#             multi_heads: [1,2,4,4,6,8,16];
#             hidden_dimension: [8,16,32,64,128,256,64];
#             activation: [sigmoid,tanh,relu6,leaky_relu,softplus,elu,linear];
#         }
#         #For convenience, the candidate value corresponding to each component is represented by a number, ranging from #0 to 6#
#         The model is a two-layer GNN model, take a model for example: the corresponding architecture of [3,1,3,5,2,6,0,3,3,1] is ['const','mean','4','256','relu6','linear','sum','4','64','tanh']
#         #### note:
#         Once again, your task is to help me find the optimal model on the dataset of ''' + self.dataname + '''. The main difficulty of this task lies in how to reasonably arrange the search strategies in different stages. We should choose a new model to try based on the existing model and its corresponding feedback, so as to iteratively find the best model.
#         At the beginning, when there were few experimental results, we in the Exploration phase, we need to explore the search space and identify which model is promising. We can #randomly# select a batch of models and evaluate their performance. Afterwards, we can sort the models based on their feedback and select some well performing models as candidates for our Exploitation phase.
#         When we have a certain amount of experimental results, we are in the Exploitation phase, we focus on improving search by exploring the search space more effectively. We can use optimization algorithms, such as Bayesian optimization or evolutionary algorithms, to focus on those models that perform best among the models that have been searched for experimental results, rather than randomly selecting them.
#