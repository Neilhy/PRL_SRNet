import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return Net(args)

class Net(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(Net, self).__init__()

        n_resblocks = 32
        n_feats = 256
        kernel_size = 3 
        scale = 4
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        x = self.head(x) # 3 x 16 x 16 -> 256 x 16 x 16

        selected_m = '15'
        selected_feature = None

        res = x
        for m_name, m in self.body._modules.items():
            res = m(res)
            if m_name == selected_m:
                selected_feature = res
        
        res += x           # 256 x 16 x 16

        x = self.tail(res) # 256 x 16 x 16 -> 256 x 64 x 64 -> 3 x 64 x 64

        return x, selected_feature