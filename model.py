import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.conv_layer import Conv1d

class FirstBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.4, norm_layer=nn.BatchNorm1d, *args, **kwargs):
        super(FirstBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.dr1 = nn.Dropout(dropout)

        # self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = norm_layer(planes)
        # self.dr2 = nn.Dropout(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )

        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)
    
    def forward(self, x, mask=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dr1(out)
        
        if mask is not None:
            out = out * mask.unsqueeze(1)

        w = F.avg_pool1d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        out = out * w
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    

class SelfAttention(nn.Module):
    def __init__(self, d_model, heads, attn_drop=0.1, res_drop=0.1):
        super().__init__()
        assert d_model % heads == 0

        self.heads = heads
        self.d_model = d_model
        # self.d_head = d_model // heads

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_drop)
        self.res_drop = nn.Dropout(res_drop)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        k = self.key(x).view(batch_size, seq_len, self.heads, embed_dim // self.heads).transpose(1, 2)
        q = self.query(x).view(batch_size, seq_len, self.heads, embed_dim // self.heads).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.heads, embed_dim // self.heads).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            att = att.masked_fill(mask == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = att.masked_fill(att.isnan(), 0.)
        att = self.attn_drop(att)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        y = self.res_drop(self.proj(y))
        return y

    
class SecondBlock(nn.Module):
    def __init__(self, num_features, *args, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(num_features)
        self.ln2 = nn.LayerNorm(num_features)
        self.attn = SelfAttention(num_features, 8, attn_drop=0.1, res_drop=0.1)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 2 * num_features),
            nn.GELU(),
            nn.Linear(2 * num_features, num_features),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x + self.mlp(self.ln2(x))
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
    
class multiscale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(multiscale, self).__init__()

        self.conv0 = Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False)

        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False, bn=False),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

        self.conv2 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
        )

        self.conv3 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True)
        )
        
    def forward(self, x):

        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = torch.cat([x0, x1, x2, x3], dim=1)
        return x4 + x        
    
class Net(nn.Module):
    def __init__(self,hidden_size=768, conv_out_channels=128, scale_out_channels = 32, num_features=64):
        super(Net, self).__init__()
        self.conv0 = Conv1d(hidden_size, conv_out_channels, kernel_size=(1,), stride=1)
        self.conv1 = Conv1d(1, conv_out_channels, kernel_size=(3,), stride=1, same_padding=False)
        self.multiscale_str = multiscale(conv_out_channels, scale_out_channels)
        self.multiscale_bert = multiscale(conv_out_channels, scale_out_channels)
        self.first_block = FirstBlock(scale_out_channels * 8, num_features)
        self.second_block = SecondBlock(num_features)
        self.fc = nn.Linear(123, 1)  # 添加全连接层
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, embedding, structure, mask=None):
        x0 = embedding  
        x1 = structure  
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x0 = self.multiscale_bert(x0)
        x1 = self.multiscale_str(x1)
        out = torch.cat([x0, x1], dim=1)
        out = self.first_block(out, mask)
        out = out.transpose(1, 2)
        out = self.second_block(out, mask)
        out = out.mean(dim=2)  # 全局平均池化
        out = out.view(out.size(0), -1)
        out = self.fc(out)  # 全连接层
        return out