import torch
import torch.nn as nn


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx[:, :, :]


class GCBlock(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(GCBlock, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel

        assert self.knn_num == 9 or self.knn_num == 6
        if self.knn_num == 9:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), #[32,128,2000,9]→[32,128,2000,3]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 3)), #[32,128,2000,3]→[32,128,2000,1]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channel*2, self.in_channel, (1, 3), stride=(1, 3)), #[32,128,2000,6]→[32,128,2000,2]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.in_channel, self.in_channel, (1, 2)), #[32,128,2000,2]→[32,128,2000,1]
                nn.BatchNorm2d(self.in_channel),
                nn.ReLU(inplace=True),
            )

    def forward(self, features):
        # feature[32,128,2000,1]
        B, _, N, _ = features.shape
        out = get_graph_feature(features, k=self.knn_num)
        out = self.conv(out)  # out[32,128,2000,1]

        return out


def get_graph_feature(x, k=20, idx=None):
    # x[32,128,2000,1],k=9
    # x[32,128,1000,1],k=6
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class EfficientAttention(nn.Module):

    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0., qkv_bias=True, k_num=9):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        qkvs, qkvs1 = [], []
        dgcnn = []
        for i in range(self.num_heads):
            qkvs.append(nn.Conv2d(dim, self.dim_head, 1, 1, 0, bias=qkv_bias))
            qkvs1.append(nn.Conv2d(dim, 3 * self.dim_head, 1, 1, 0, bias=qkv_bias))
            dgcnn.append(GCBlock(k_num, in_channel=self.dim_head))

        self.qkvs = nn.ModuleList(qkvs)
        self.qkvs1 = nn.ModuleList(qkvs1)
        self.dgcnn = nn.ModuleList(dgcnn)
        self.proj_res = nn.Conv2d(dim*2, dim, 1, 1, 0, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.clusters = 50
        self.conv_group1 = nn.Sequential(
            nn.InstanceNorm2d(dim, eps=1e-3),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, self.clusters, kernel_size=1)
        )
        self.conv_group2 = nn.Sequential(
            nn.InstanceNorm2d(dim, eps=1e-3),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, self.clusters, kernel_size=1)
        )

    def local_conv(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module):
        '''
        x: (b c h w)  self.qkvs[i], self.convs[i], self.act_blocks[i]
        '''
        b, c, n, _ = x.size()
        qk = to_qkv(x)  # (b d n 1)
        qk = qk.reshape(b, self.dim_head, self.clusters, 1).contiguous()
        qk = mixer(qk)
        return qk

    def global_attention(self, x: torch.Tensor, to_qkv: nn.Module):
        '''
        x: (b c n 1) to_q:global_q to_kv :global_kv
        '''
        b, c, n, _ = x.size()
        qkv = to_qkv(x)
        qkv = qkv.reshape(b, 3, -1, self.clusters).transpose(0, 1).contiguous()
        q, k, v = qkv
        attn = self.scalor * q.transpose(-1, -2) @ k  # (b n n)
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v.transpose(-1, -2)  # (b n d)
        res = res.transpose(-1, -2).reshape(b, -1, self.clusters, 1).contiguous()

        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c n 1)
        '''
        embed1 = self.conv_group1(x)
        S1 = torch.softmax(embed1, dim=2).squeeze(3)
        cluster_x = torch.matmul(x.squeeze(3), S1.transpose(1, 2)).unsqueeze(3)

        res = []
        for i in range(self.num_heads):
            res.append(self.local_conv(cluster_x, self.qkvs[i], self.dgcnn[i]))
            res.append(self.global_attention(cluster_x, self.qkvs1[i]))

        res = self.proj_res(torch.cat(res, dim=1))

        embed2 = self.conv_group2(x)
        S2 = torch.softmax(embed2, dim=1).squeeze(3)
        res = torch.matmul(res.squeeze(3), S2).unsqueeze(3)

        return self.proj_drop(self.proj(res))


class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DATBlock(nn.Module):

    def __init__(self, dim, out_dim, num_heads, mlp_ratio: int, k_num, attn_drop=0., mlp_drop=0., qkv_bias=True,
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = EfficientAttention(dim, num_heads, attn_drop, mlp_drop, qkv_bias, k_num)
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(dim, mlp_hidden_dim, out_dim, drop_out=mlp_drop)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

