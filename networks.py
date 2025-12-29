from calendar import c
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resume import ResUME
from modules import Encoder_x2, Decoder_x2, ChannelEncoder, ChannelDecoder
from mobilenet import MobileNetV2Encoder, MobileNetV2Decoder
from mobilevit import MobileViTEncoder, MobileViTDecoder

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
        snr_fusion = torch.clip(snr[0], 0.0, 15.0)
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


class Analog_SemCom(nn.Module):
    """
    Digital_SemCom과 동일한 encoder/decoder 구조를 사용하지만,
    ResUME 기반의 RVQ + QAM mapping 없이 latent를 그대로
    연속값 채널(AWGN + optional fading)을 통과시키는 아날로그 SemCom 모델.
    bottleneck dimension은 embedding_dim으로 Digital과 동일하게 유지됨.
    """
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                embedding_dim, target_CBR,):
        # num_stages, vq_bitrate_per_stage, batch_size, device는
        # Digital_SemCom과 동일한 인터페이스 유지를 위해 받지만 여기서는 사용하지 않음.
        super(Analog_SemCom, self).__init__()
        
        # --- Encoder / bottleneck 부분: Digital과 동일 --- #
        self._encoder = Encoder_x2(
            in_channels=3,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self._pre_vq_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=embedding_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        # SNR-adaptive gating (ChannelEncoder/Decoder)는 그대로 사용
        self.noise_fusion_enc = ChannelEncoder(
            len_feature=embedding_dim, num_fusion=7, residue_len=2
        )
        self.noise_fusion_dec = ChannelDecoder(
            len_feature=embedding_dim, num_fusion=7, residue_len=2
        )
        bottleneck_dim = int(target_CBR * 3 * 128 * 128 // 64)
        print(f"[Info] Analog_SemCom bottleneck dimension: {bottleneck_dim}")
        self.bottleneck_enc = nn.Linear(embedding_dim, bottleneck_dim)
        self.bottleneck_dec = nn.Linear(bottleneck_dim, embedding_dim)

        # --- Decoder: Digital과 동일 --- #
        self._decoder = Decoder_x2(
            in_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def analog_channel(self, x, snr_db, apply_fading=False, block_fading=True):
        """
        연속값 latent 벡터 x에 AWGN(+선택적 페이딩)을 적용하는 아날로그 채널.

        x      : [N, D] 형태의 latent
        snr_db : torch.Tensor 또는 리스트 형태, Digital과 동일하게 snr_db[0] 사용
        """
        # 평균 전력 기준 SNR에 맞춰 노이즈 분산 설정
        snr_linear = 10 ** (snr_db[0] / 10)
        power = x.pow(2).mean()                     # 평균 신호 전력
        noise_var = power / (snr_linear + 1e-12)    # σ^2
        noise_std = torch.sqrt(noise_var + 1e-12)
        noise = torch.randn_like(x) * noise_std

        if apply_fading:
            # block_fading이면 샘플마다 하나의 페이딩 계수, 아니면 element-wise 계수
            if block_fading:
                h_shape = (x.size(0), 1)
            else:
                h_shape = x.shape

            # 실수 도메인 페이딩 계수 (N(0,1)); 평균 전력 기준 Rayleigh와 유사한 스케일
            h = torch.randn(h_shape, device=x.device)

            y = x * h + noise  # 페이딩 + 노이즈
            # 간단한 ZF equalization: y / h
            y_eq = y / (h + 1e-8)
            return y_eq
        else:
            # pure AWGN channel
            return x + noise

    def forward(
        self,
        x,
        sum_stage,
        snr,
        m_idx,
        apply_fading,
        rvq_activate=True,
        nsvq=True,
        equalizer="zf",
    ):
        """
        Digital_SemCom과 동일한 인터페이스를 유지하기 위해
        sum_stage, m_idx, rvq_activate, nsvq, equalizer 인자를 받지만
        아날로그 모드에서는 quantization / modulation을 수행하지 않음.
        """
        # ---------------- Encoder ---------------- #
        z = self._encoder(x)
        z = self._pre_vq_conv(z)          # [B, C=embedding_dim, H', W']
        z = z.permute(0, 2, 3, 1)         # [B, H', W', C]
        origin_shape = z.shape
        z = z.reshape(-1, z.shape[-1])    # [B * H' * W', embedding_dim]

        # snr_fusion은 Digital_SemCom과 동일하게 clipping
        snr_fusion = torch.clip(snr[0], 0.0, 10.0)
        z_e = self.noise_fusion_enc(z, snr_fusion)  # encoder-side SNR gating
        z_e = self.bottleneck_enc(z_e)              # bottleneck 차원 축소

        # ---------------- Analog Channel ---------------- #
        # 여기서는 ResUME(VQ)나 bit-packing/QAM mapping 없이
        # 연속값 latent를 그대로 채널 노이즈를 통과시킴.
        z_noisy = self.analog_channel(z_e, snr, apply_fading=apply_fading)

        # ---------------- Decoder-side fusion + Decoder ---------------- #
        z_noisy_expand = self.bottleneck_dec(z_noisy)      # bottleneck 차원 복원
        quantized = self.noise_fusion_dec(z_noisy_expand, snr_fusion)
        quantized = quantized.reshape(origin_shape).permute(0, 3, 1, 2)
        x_recon = self._decoder(quantized)

        # Digital_SemCom과 동일한 반환 형태를 맞추기 위해 placeholder 리턴
        used_codebook = None
        perplexity = None
        loss = torch.tensor(0.0, device=x.device)

        # z_q 자리에는 채널을 통과한 연속 latent인 z_noisy를 넣어 둠
        return x_recon, used_codebook, perplexity, z_e, z_noisy, loss


class MobileNet_SemCom(nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, embedding_dim, batch_size, device):
        super(MobileNet_SemCom, self).__init__()
        self._encoder = MobileNetV2Encoder(ch_in=3, width_mult=1.0, use_bn=True, latent_ch=embedding_dim)
        self.noise_fusion_enc = ChannelEncoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.noise_fusion_dec = ChannelDecoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.resume = ResUME(num_stages, vq_bitrate_per_stage, data_dim=embedding_dim, batch_size=batch_size, device=device)
        self._decoder = MobileNetV2Decoder(in_ch=embedding_dim, out_ch=3, width_mult=1.0, use_bn=True)

    def forward(self, x, sum_stage, snr, m_idx, apply_fading, rvq_activate=True, nsvq=True, equalizer='zf'):
        z = self._encoder(x)
        z = z.permute(0, 2, 3, 1)
        origin_shape = z.shape
        z = z.reshape(-1, z.shape[-1]) ## [batch_size * 64, 128]
        snr_fusion = torch.clip(snr[0], 0.0, 15.0)
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
    
class MobileViT_SemCom(nn.Module):
    def __init__(self, num_stages, vq_bitrate_per_stage, embedding_dim, batch_size, device, variant="xs"):
        super(MobileViT_SemCom, self).__init__()
        self.noise_fusion_enc = ChannelEncoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.noise_fusion_dec = ChannelDecoder(len_feature=embedding_dim, num_fusion=7, residue_len=2)
        self.resume = ResUME(num_stages, vq_bitrate_per_stage, data_dim=embedding_dim, batch_size=batch_size, device=device)

        assert variant in ["xxs","xs","s"]
        # Configs chosen to be consistent & simple
        if variant == "xxs":
            cfg = dict(
                # channels
                c0=16, c1=16, c2=24, c3=48, c4=64, c5=80, cb=160,
                # ViT dims & depths
                d1=64, d2=80, d3=96, L1=2, L2=4, L3=3,
            )
            expansion = 2
        elif variant == "xs":
            cfg = dict(
                c0=16, c1=32, c2=48, c3=64, c4=80, c5=128, cb=192,
                d1=96, d2=120, d3=144, L1=2, L2=4, L3=3,
            )
            expansion = 4
        else:  # "s"
            cfg = dict(
                c0=16, c1=32, c2=64, c3=96, c4=128, c5=128, cb=320,
                d1=144, d2=192, d3=240, L1=2, L2=4, L3=3,
            )
            expansion = 4

        self._encoder = MobileViTEncoder(
            image_size=(128,128), cfg=cfg, expansion=expansion, kernel_size=3, patch_size=(2,2)
        )
        self._decoder = MobileViTDecoder(
            image_size=(128,128), cfg=cfg, kernel_size=3, patch_size=(2,2)
        )
    def forward(self, x, sum_stage, snr, m_idx, apply_fading, rvq_activate=True, nsvq=True, equalizer='zf'):
        z = self._encoder(x)
        z = z.permute(0, 2, 3, 1)
        origin_shape = z.shape
        z = z.reshape(-1, z.shape[-1]) ## [batch_size * 64, 128]
        snr_fusion = torch.clip(snr[0], 0.0, 15.0)
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
    

# -------------------- quick test --------------------


# if __name__ == "__main__":
#     from profiling_utils import profile_one_model, pretty_print_profiles
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     x = torch.randn(1, 3, 128, 128, device=device)

#     our_model = Digital_SemCom(
#         num_hiddens=128, num_residual_layers=4, num_residual_hiddens=64,
#         num_stages=4, vq_bitrate_per_stage=12, embedding_dim=128,
#         batch_size=1, device=str(device)
#     ).to(device)

#     analog_model = Analog_SemCom(
#         num_hiddens=128, num_residual_layers=4, num_residual_hiddens=64,
#         embedding_dim=128, target_CBR=0.00521
#     ).to(device)

#     mobilenet_model = MobileNet_SemCom(
#         num_stages=4, vq_bitrate_per_stage=12, embedding_dim=128,
#         batch_size=1, device=str(device)
#     ).to(device)

#     mobilevit_model = MobileViT_SemCom(
#         num_stages=4, vq_bitrate_per_stage=12, embedding_dim=128,
#         batch_size=1, device=str(device)
#     ).to(device)

#     # Use fixed forward kwargs for fair comparison
#     fwd_kwargs = dict(
#         sum_stage=4,
#         snr=torch.tensor([10.0], device=device),
#         m_idx=[0, 1, 2, 3],
#         apply_fading=False,
#         rvq_activate=True,
#         nsvq=True,
#         equalizer="zf",
#     )

#     profiles = []
#     profiles.append(profile_one_model("Digital_SemCom", our_model, (x,), fwd_kwargs))
#     profiles.append(profile_one_model("Analog_SemCom", analog_model, (x,), fwd_kwargs))
#     profiles.append(profile_one_model("MobileNet_SemCom", mobilenet_model, (x,), fwd_kwargs))
#     profiles.append(profile_one_model("MobileViT_SemCom", mobilevit_model, (x,), fwd_kwargs))

#     pretty_print_profiles(profiles)

if __name__ == "__main__":
    from profiling_utils import profile_one_model, pretty_print_profiles, find_max_batch, build_fwd_kwargs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ 여기서 threshold만 네 GPU 상황에 맞게 설정하면 됨 (MiB 단위)
    # 예: 8000 = 8GB, 12000 = 12GB ...
    MAX_MEM_MIB = 12000.0

    # 어떤 메모리를 기준으로 제한할지:
    # - reserved: 더 보수적(캐시/워크스페이스 포함)
    # - allocated: 실제 할당 기준(조금 덜 보수적)
    USE_RESERVED = False

    # 모델 생성 함수를 "batch_size -> model" 형태로 통일
    def build_digital(bs):
        return Digital_SemCom(
            num_hiddens=128, num_residual_layers=4, num_residual_hiddens=64,
            num_stages=4, vq_bitrate_per_stage=12, embedding_dim=128,
            batch_size=bs, device=str(device)
        )

    def build_analog(bs):
        # Analog_SemCom 생성자에 batch_size가 없어서 그대로 두되,
        # 입력 x 배치만 바꿔서 forward는 bs로 실행됨.
        return Analog_SemCom(
            num_hiddens=128, num_residual_layers=4, num_residual_hiddens=64,
            embedding_dim=128, target_CBR=0.00521
        )

    def build_mobilenet(bs):
        return MobileNet_SemCom(
            num_stages=4, vq_bitrate_per_stage=12, embedding_dim=128,
            batch_size=bs, device=str(device)
        )

    def build_mobilevit(bs):
        return MobileViT_SemCom(
            num_stages=4, vq_bitrate_per_stage=12, embedding_dim=128,
            batch_size=bs, device=str(device)
        )

    model_builders = [
        ("Analog_SemCom", build_analog),
        ("Digital_SemCom", build_digital),
        ("MobileNet_SemCom", build_mobilenet),
        ("MobileViT_SemCom", build_mobilevit),
    ]

    results = []

    for name, builder in model_builders:
        # 1) 최대 배치 찾기
        max_bs, peak_mib = find_max_batch(
            builder, device=device,
            max_mem_mib=MAX_MEM_MIB,
            start_bs=1, max_bs_cap=4096,
            use_reserved=USE_RESERVED
        )

        if max_bs <= 0:
            print(f"[{name}] batch_size=1도 threshold({MAX_MEM_MIB} MiB) 이내로 못 들어옴 (peak ~ {peak_mib:.1f} MiB)")
            continue

        # 2) 그 배치에서 latency만 프로파일링
        model = builder(max_bs).to(device)
        x = torch.randn(max_bs, 3, 128, 128, device=device)
        fwd_kwargs = build_fwd_kwargs(device, max_bs)

        prof = profile_one_model(name, model, (x,), fwd_kwargs)

        # ✅ "latency만" + 참고로 max batch 기록
        # (pretty_print_profiles가 어떤 키를 쓰는지 모르니, 최소 키만 보강)
        prof["MaxBatch"] = max_bs
        prof["PeakMem(MiB)@Search"] = peak_mib

        # 필요하면 다른 값들은 지워도 됨. 여기서는 latency 위주로 출력만 별도 제공.
        results.append(prof)

    pretty_print_profiles(results)

    print("\n--- Max batch under memory threshold + Latency ---")
    print(f"Threshold: {MAX_MEM_MIB:.0f} MiB  |  criterion: {'reserved' if USE_RESERVED else 'allocated'}")
    for r in results:
        name = r.get("name", r.get("Model", "MODEL"))
        lat = r.get("Lat(ms)", r.get("lat_mean_ms", None))
        print(f"{name:16s}  MaxBatch={r['MaxBatch']:4d}  Lat(ms)={lat}  PeakMem@Search={r['PeakMem(MiB)@Search']:.1f} MiB")