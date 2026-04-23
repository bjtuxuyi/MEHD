import pickle
import numpy as np
import random
import torch
import torch.nn  as nn
import torch.sparse  as tsp


class SDHGCN(nn.Module):
    def __init__(self, in_features, out_features, use_norm=True):
        super().__init__()
        self.weight  = nn.Parameter(torch.Tensor(in_features, out_features))
        self.use_norm  = use_norm
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, X, adj_matrix):
        H_in, H_out = self.build_hyper_edges(adj_matrix)

        # 特征传播
        support = tsp.mm(H_in.t(),  X)
        transformed = torch.mm(support,  self.weight)
        output = tsp.mm(H_out,  transformed)

        # 归一化处理
        if self.use_norm:
            norm_coeff = self.calculate_norm(adj_matrix)
            output = output * norm_coeff.unsqueeze(1)

        return torch.relu(output)

    def build_hyper_edges(self, adj):
        adj = adj.to(torch.bool)

        in_rows, in_cols = torch.where(adj)
        in_rows = in_rows.to(device=adj.device,  dtype=torch.long)   # 指定设备+类型
        in_cols = in_cols.to(device=adj.device,  dtype=torch.long)

        # 构造H_in
        H_in_idx = torch.stack([in_rows,  in_cols], dim=0).to(torch.long)
        H_in = tsp.FloatTensor(
            indices=H_in_idx,
            values=torch.ones(in_rows.size(0),  dtype=torch.float32),
            size=(adj.size(0),  adj.size(1))
        )

        # 构造H_out
        out_idx = torch.stack([
            torch.arange(adj.size(0),  dtype=torch.long,  device=adj.device),
            torch.arange(adj.size(0),  dtype=torch.long,  device=adj.device)
        ])
        H_out = tsp.FloatTensor(
            indices=out_idx,
            values=torch.ones(adj.size(0),  dtype=torch.float32),
            size=(adj.size(0),  adj.size(1))
        )

        return H_in.coalesce(),  H_out.coalesce()

    def calculate_norm(self, adj):
        # 计算归一化系数 (基于节点入度)
        in_degree = adj.sum(dim=0).float()   # [N]
        norm = torch.pow(in_degree.clamp(min=1),  -0.5)
        return norm.to(adj.device)

if __name__ == '__main__':
    # 参数设置
    N = 1024  # 节点数
    F_in = 64  # 输入特征维度
    F_out = 128  # 输出特征维度

    adj = (torch.rand(N, N) < 0.1).long()  # 生成int64类型矩阵
    features = torch.randn(N, F_in)

    print(type(features))
    print(type(adj))
    print(adj.shape)

    conv_layer = SDHGCN(F_in, F_out)

    # 前向传播
    output = conv_layer(features, adj)
    print(output.shape)  # torch.Size([1024, 128])
    print(type(output))  # torch.Size([1024, 128])
