import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resume import ResUME
from modules import Encoder_x2, Decoder_x2, ChannelEncoder, ChannelDecoder


class Digital_SemCom(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_stages, vq_bitrate_per_stage, embedding_dim, batch_size, device):
        super(Digital_SemCom, self).__init__()
        
        self._encoder = Encoder_x2(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_hiddens, 
                        out_channels=embedding_dim,
                        kernel_size=4, 
                        stride=2,
                        padding=1),   
        )
        self.noise_fusion_enc = ChannelEncoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.noise_fusion_dec = ChannelDecoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.resume = ResUME(num_stages, vq_bitrate_per_stage, data_dim=embedding_dim, batch_size=batch_size, device=device)

        self._decoder = Decoder_x2(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x, sum_stage, snr, m_idx, apply_fading, rvq_activate=True, nsvq=True, equalizer='zf'):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        z = z.permute(0, 2, 3, 1)
        origin_shape = z.shape
        z = z.reshape(-1, z.shape[-1]) ## [batch_size * 64, 128]
        snr_fusion = torch.clip(snr[0], 0.0, 10.0)
        z_e = self.noise_fusion_enc(z, snr_fusion)
        if rvq_activate == True:
            z_q, used_codebook, perplexity, loss = self.resume(z_e, sum_stage, snr, m_idx, nsvq=nsvq, apply_fading=apply_fading, equalizer=equalizer)
        else:
            z_q = z_e
            used_codebook, perplexity = None, None
        quantized = self.noise_fusion_dec(z_q, snr_fusion)
        quantized = quantized.reshape(origin_shape).permute(0, 3, 1, 2)
        x_recon = self._decoder(quantized)

        return x_recon, used_codebook, perplexity, z_e, z_q, loss