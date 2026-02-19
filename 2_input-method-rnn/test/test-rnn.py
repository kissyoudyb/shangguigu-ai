import torch
from torch import nn

rnn = nn.RNN(
    input_size=3,        # 输入特征维度：每个时间步的输入向量长度为3
    hidden_size=4,       # 隐藏层维度：每个RNN单元的输出/隐藏状态维度为4
    batch_first=True,    # 输入数据格式为 [batch_size, seq_len, input_size]（默认是[seq_len, batch_size, input_size]）
    num_layers=2,        # 网络层数：2层RNN堆叠
    bidirectional=True   # 双向RNN：同时从序列开头→结尾和结尾→开头两个方向计算
)

# input.shape: [batch_size, seq_len, input_size]
input = torch.randn(2, 4, 3)
# 含义：批量大小=2（2个样本），序列长度=4（每个样本有4个时间步），输入维度=3（与input_size一致）

# output.shape: [batch_size, seq_len, 2*hidden_size]
# hn.shape: [num_layers × num_directions, batch_size, hidden_size]
output, hn = rnn(input)
# output：所有时间步的输出（最后一层）
# hn：最后一个时间步的隐藏状态（所有层）

print(output.shape)
print(hn.shape)
