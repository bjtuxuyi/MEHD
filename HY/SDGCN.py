import torch
from torch_sparse import SparseTensor
import numpy as np

def dense_to_coo(dense_matrix):
    rows, cols = np.where(dense_matrix != 0)
    row_tensor = torch.from_numpy(rows.astype(np.int64)).long()
    col_tensor = torch.from_numpy(cols.astype(np.int64)).long()
    data = torch.from_numpy(dense_matrix[rows, cols]).float()
    return torch.sparse_coo_tensor(
        indices=torch.stack([row_tensor, col_tensor]), values=data, size=dense_matrix.shape
    )
class SparseDirectionEncoder:
    def __init__(self, edge_index, num_nodes):
        # 分离出入度矩阵
        adj_out = SparseTensor(row=edge_index.coalesce().indices()[0],col=edge_index.coalesce().indices()[1],sparse_sizes=(num_nodes, num_nodes))
        adj_in = adj_out.t()
        # 正则化
        deg_out = adj_out.sum(dim=1).pow(-0.5)
        deg_in = adj_in.sum(dim=1).pow(-0.5)
        self.adj_norm = deg_out.view(-1, 1) * adj_out * deg_in.view(1, -1)
class DirectedGraphConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim,device):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.Tensor([0.5])) # 方向权重
        self.beta = torch.nn.Parameter(torch.Tensor([0.5]))
        self.W = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.W_dir = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.device = device
    def forward(self, x, adj_norm):
        transformed = self.W(x)
        adj_norm = adj_norm.to(self.device)
        # print(adj_norm.device)
        # print(transformed.device)
        if isinstance(adj_norm, SparseTensor): h_base = adj_norm.matmul(transformed)
        else: h_base = torch.matmul(adj_norm, transformed)
        # print("self.alpha.shape:", self.alpha.shape)
        # print("x.shape:", x.shape)
        # print("self.beta.shape:", self.beta.shape)
        # print("self.W_dir(x).shape:", self.W_dir(x).shape)
        h_dir = self.alpha * transformed + self.beta * self.W_dir(x)
        return h_base + h_dir
class SDGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, num_nodes,device):
        super().__init__()
        self.encoder = SparseDirectionEncoder(edge_index, num_nodes)
        self.conv1 = DirectedGraphConv(in_dim, hidden_dim,device)
        self.conv2 = DirectedGraphConv(hidden_dim, out_dim,device)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        x = self.conv1(x, self.encoder.adj_norm)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, self.encoder.adj_norm)
        return x


if __name__ == '__main__':
    EventDirectAdjacencyMatrix = np.load("../dataset/Jinghu_HG/EventDirectAdjacencyMatrix.npy")
    print(f"事件图有向邻接矩阵维度：{EventDirectAdjacencyMatrix.shape}, 数据类型：{EventDirectAdjacencyMatrix.dtype},存在为1，不存在为0")

