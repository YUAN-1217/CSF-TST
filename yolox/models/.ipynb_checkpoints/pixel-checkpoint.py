import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # print("Before unfold", x.shape)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # print("After unfold", x.shape)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        # print(x.shape, kernel.shape)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

class PicanetL_light(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL_light, self).__init__()
        # 将7x7卷积分解为两个3x3卷积
        self.conv1_1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, dilation=2, padding=2)
        # 减少中间通道数
        self.conv2 = nn.Conv2d(128, 36, kernel_size=1) # 将49减少到36
        
    def forward(self, x):
        size = x.size()
        kernel = self.conv1_1(x)
        kernel = self.conv1_2(kernel)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 6 * 6)  # 根据新的通道数调整
        x = F.unfold(x, kernel_size=[6, 6], dilation=[2, 2], padding=5)  # 调整卷积核大小与填充
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

class ModernPicanetL(nn.Module):
    def __init__(self, in_channel):
        super(ModernPicanetL, self).__init__()
        
        # 特征提取模块 - 使用BatchNorm2d替代LayerNorm
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 增强的局部注意力模块
        self.local_attention = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6, groups=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 49, kernel_size=1),
            nn.Dropout(0.1)
        )
        
        # 输出特征增强
        self.output_enhance = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.GELU()
        )

    def forward(self, x):
        identity = x
        size = x.size()
        
        # 1. 特征增强
        x = self.feature_extract(x)
        
        # 2. 生成局部注意力权重
        kernel = self.local_attention(x)
        kernel = F.softmax(kernel, dim=1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        
        # 3. 应用局部注意力
        patches = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        patches = patches.reshape(size[0], size[1], size[2] * size[3], -1)
        weighted = torch.mul(patches, kernel)
        weighted = torch.sum(weighted, dim=3)
        out = weighted.reshape(size[0], size[1], size[2], size[3])
        
        # 4. 特征增强和残差连接
        out = self.output_enhance(out)
        out = out + identity
        
        return out

class PicanetG(nn.Module):

    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 100)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)
        x = F.unfold(x, [10, 10], dilation=[3, 3])
        x = x.reshape(size[0], size[1], 10 * 10)
        kernel = kernel.reshape(size[0], 100, -1)
        x = torch.matmul(x, kernel)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)

    def forward(self, *input):
        x = input[0]
        temp = []
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = self.conv(x)
        return x


# class ModernPicanetG(nn.Module):
#     def __init__(self, size, in_channel):
#         super(ModernPicanetG, self).__init__()
#         self.size = size
#         self.in_channel = in_channel
        
#         # 使用自注意力替代LSTM
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=in_channel,
#             num_heads=8,
#             dropout=0.1
#         )
        
#         # 添加层归一化
#         self.norm1 = nn.LayerNorm(in_channel)
#         self.norm2 = nn.LayerNorm(in_channel)
        
#         # 使用前馈网络
#         self.ffn = nn.Sequential(
#             nn.Linear(in_channel, in_channel * 4),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(in_channel * 4, in_channel)
#         )
        
#         # 输出投影
#         self.out_proj = nn.Conv2d(in_channel, 100, 1)

#     def forward(self, x):
#         b, c, h, w = x.size()
        
#         # 重塑输入以适应自注意力
#         x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
#         # 自注意力 + 残差
#         attn_out, _ = self.self_attn(x, x, x)
#         x = x + attn_out
#         x = self.norm1(x)
        
#         # FFN + 残差
#         ffn_out = self.ffn(x)
#         x = x + ffn_out
#         x = self.norm2(x)
        
#         # 重塑回空间维度
#         x = x.transpose(1, 2).reshape(b, c, h, w)
        
#         # 输出投影
#         kernel = self.out_proj(x)
#         kernel = F.softmax(kernel, dim=1)
        
#         # 应用注意力
#         x = F.unfold(x, [10, 10], dilation=[3, 3])
#         x = x.reshape(b, c, 100, -1)
#         x = torch.matmul(x.transpose(1, 2), kernel.view(b, 100, -1))
#         x = x.reshape(b, c, h, w)
        
#         return x
class ModernPicanetG(nn.Module):   #效果差
    def __init__(self, size, in_channel):
        super(ModernPicanetG, self).__init__()
        self.size = size
        self.in_channel = in_channel
        
        # 简化注意力机制
        self.self_attn = nn.MultiheadAttention(
            embed_dim=in_channel,
            num_heads=8,
            dropout=0.1
        )
        
        self.norm = nn.LayerNorm(in_channel)
        
        # 简化FFN
        self.ffn = nn.Sequential(
            nn.Linear(in_channel, in_channel * 2),
            nn.ReLU(),
            nn.Linear(in_channel * 2, in_channel)
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 自注意力
        identity = x
        x_flat = x.flatten(2).transpose(1, 2)
        attn_out, _ = self.self_attn(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = self.norm(x_flat)
        
        # FFN
        x_ffn = self.ffn(x_flat)
        x_flat = x_flat + x_ffn
        
        # 重塑
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        
        # 空间注意力
        spatial_weight = self.spatial_attn(x)
        x = x * spatial_weight + identity
        
        return x
# 主要改进包括:

# 用现代的多头自注意力机制替代LSTM结构
# 添加了层归一化(LayerNorm)提高训练稳定性
# 使用残差连接避免梯度消失
# 添加dropout增强泛化能力
# 使用更简洁的张量操作替代循环操作
# 添加了前馈网络(FFN)增强特征转换能力
# 这些改进的优势:

# 更好的并行计算效率
# 更强的特征提取能力
# 更稳定的训练过程
# 更好的长程依赖建模能力
# 更现代的架构设计
# 你可以根据具体任务需求调整:

# 注意力头数(num_heads)
# dropout率
# 网络宽度(in_channel * 4)
# 输出通道数(100)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class PicanetGimprove(nn.Module):  #修改了现在基本上就是纯自注意力机制了
    def __init__(self, size, in_channel):
        super(PicanetGimprove, self).__init__()
        self.attention = SelfAttention(in_channel, in_channel)
        self.in_channel = in_channel
        self.size = size
        self.kernel_size = min(3, size)

    def forward(self, x):
        size = x.size()  # [B, C, H, W]
        B, C, H, W = size
        
        # 1. 只使用自注意力，简化模型
        identity = x
        attention_out = self.attention(x)  # [B, C, H, W]
        
        # 2. 应用全局上下文
        x_global = attention_out
        
        # 3. 融合局部和全局特征
        out = x_global + identity
        
        return out
# SelfAttention 模块：

# 使用自注意力机制来生成注意力权重，替代了原来的 LSTM 结构。
# query, key, value 三个卷积层分别用于生成查询、键和值。
# 通过矩阵乘法和 softmax 计算注意力权重，并将其应用于输入特征图。
# PicanetG 模块：

# 使用 SelfAttention 模块生成注意力权重。
# 保留了原有的 F.unfold 和 torch.matmul 操作，但结构更加简洁高效。
# PicanetL 模块：

# 保持不变，因为它已经相对简单且高效。
# 通过这些优化，模型的计算效率和性能应该会有所提升

class SimplifiedPicanetL(nn.Module):
    def __init__(self, in_channel):
        super(SimplifiedPicanetL, self).__init__()
        
        # 优化的主干注意力分支
        self.attention = nn.Sequential(
            # 使用分组卷积提高效率
            nn.Conv2d(in_channel, 128, 
                     kernel_size=7, 
                     dilation=2, 
                     padding=6, 
                     groups=4),  # 分组卷积减少计算量
            nn.BatchNorm2d(128),  # 归一化稳定训练
            nn.SiLU(),  # 更好的非线性特征
            nn.Conv2d(128, 49, kernel_size=1)
        )
        
        # 可学习的特征融合参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, *input):
        x = input[0]
        identity = x
        size = x.size()
        
        # 1. 优化的注意力计算
        kernel = self.attention(x)
        kernel = F.softmax(kernel, dim=1)  # 归一化权重
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        
        # 2. 高效的特征处理
        patches = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        patches = patches.reshape(size[0], size[1], size[2] * size[3], -1)
        
        # 3. 注意力应用
        weighted = torch.mul(patches, kernel)
        weighted = torch.sum(weighted, dim=3)
        out = weighted.reshape(size[0], size[1], size[2], size[3])
        
        # 4. 自适应特征融合
        out = identity * (1 + self.beta) + out * self.gamma
        
        return out