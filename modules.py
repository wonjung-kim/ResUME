import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder_x2(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder_x2, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)
    

class Decoder_x2(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder_x2, self).__init__()

        self._conv_1 = nn.ConvTranspose2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)
        x = F.relu(x)
        x = self._conv_trans_3(x)
        return x

class NoiseFusion(nn.Module):
    def __init__(self, len_feature):
        super(NoiseFusion, self).__init__()
        self.fc_feature = nn.Linear(len_feature, len_feature)
        self.fc1 = nn.Linear(len_feature, len_feature // 2)
        self.fc2 = nn.Linear(len_feature // 2, len_feature // 4)
        self.fc3 = nn.Linear(len_feature // 4, len_feature)

    def forward(self, x, snr):
        snr_vec = snr * torch.ones(x.shape).to(x.device)
        x_aft1 = self.fc_feature(x)
        x_aft2 =  self.fc3(self.fc2(self.fc1(snr_vec)))
        return x_aft1 * x_aft2
    
class ChannelEncoder(nn.Module):
    def __init__(self, len_feature, num_fusion, residue_len=2):
        super(ChannelEncoder, self).__init__()
        self.num_fusion = num_fusion
        self.residue_len = residue_len
        self.sigmoid = nn.Sigmoid()
        self.post_fc = nn.Linear(len_feature, len_feature)
        self.fusion_list = nn.ModuleList()
        for i in range(num_fusion):
            self.fusion_list.append(NoiseFusion(len_feature))
        
    def forward(self, x, snr):
        x_init = x
        save_list = []
        for _, layer in enumerate(self.fusion_list):
            x = layer(x, snr)
            save_list.insert(0, x)
            if len(save_list) == self.residue_len + 1:
                residue = save_list.pop(-1)
                x = x + residue
        x = self.post_fc(x)
        x = self.sigmoid(x)
        x_out = x_init * x
        return x_out
    
class ChannelDecoder(nn.Module):
    def __init__(self, len_feature, num_fusion, residue_len=2):
        super(ChannelDecoder, self).__init__()
        self.num_fusion = num_fusion
        self.residue_len = residue_len
        self.sigmoid = nn.Sigmoid()
        self.post_fc = nn.Linear(len_feature, len_feature)
        self.fusion_list = nn.ModuleList()
        for i in range(num_fusion):
            self.fusion_list.append(NoiseFusion(len_feature))
        
    def forward(self, x, snr):
        x_init = x
        save_list = []
        for _, layer in enumerate(self.fusion_list):
            x = layer(x, snr)
            save_list.insert(0, x)
            if len(save_list) == self.residue_len + 1:
                residue = save_list.pop(-1)
                x = x + residue
        x = self.post_fc(x)
        x = self.sigmoid(x)
        x_out = x_init * x
        return x_out