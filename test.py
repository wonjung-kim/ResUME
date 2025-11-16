import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import pandas as pd
from pytorch_msssim import ms_ssim
import lpips  # pip install lpips

from dataloader import ImageNet_Loader, Kodak_Patch_Loader   # <- 여기만 test용 dataloader로 바꿔도 됨
from networks import Digital_SemCom

# -------------------- 기본 설정 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

parser = argparse.ArgumentParser(description="Test Digital_SemCom on new dataset")
parser.add_argument("--ckpt", type=str, required=True,
                    help="Path to trained checkpoint (.pt)")
parser.add_argument("--model", type=str, default="gauss",
                    help="Model name (for logging paths)")
parser.add_argument("--stages", type=int, default=4,
                    help="Number of RVQ stages (must match training)")
parser.add_argument("--bits", type=int, default=12,
                    help="RVQ codeword bits (must match training)")
parser.add_argument("--batch", type=int, default=36,
                    help="Test batch size")
parser.add_argument("--norm", action="store_true",
                    help="Use same normalization as training")
parser.add_argument("--m", type=str, default="0123",
                    help="Modulation index sequence (e.g., '0123')")
parser.add_argument("--img_size", type=int, default=128,
                    help="Test image size (must match training/model)")
parser.add_argument("--dataset", type=str, default="ImageNet",
                    help="Dataset name (for your own use)")
parser.add_argument("--out_dir", type=str, default="./test_output",
                    help="Directory to save test images and metrics")
args = parser.parse_args()

# -------------------- 테스트 설정 --------------------
# SNR: -5 dB ~ 20 dB, 1 dB 간격으로 테스트
SNR_LIST = list(range(-5, 21, 1))  # [-5, -4, ..., 19, 20]

# 이미지 저장은 이 SNR들에서만
SAVE_SNR_LIST = [-5, 0, 5, 10, 15, 20]

STAGES = list(range(1, args.stages + 1))

# fading / 채널 모드 정의
# ResUME.forward 내부에서 apply_fading, equalizer를 받는 구조라고 가정
CHANNEL_MODES = [
    ("awgn",        False, "zf"),    # AWGN (no fading)
    ("rayleigh",    True,  "zf"),    # Rayleigh + ZF
]

# LPIPS 네트워크
lpips_fn = lpips.LPIPS(net="vgg").to(device)
lpips_fn.eval()

# -------------------- 모델 / 데이터 로딩 --------------------
vq_bitrate_per_stage = args.bits
embedding_dim = 128  # 학습 때 사용한 값과 동일해야 함

model = Digital_SemCom(
    num_hiddens=128,
    num_residual_hiddens=64,
    num_residual_layers=4,
    num_stages=args.stages,
    vq_bitrate_per_stage=vq_bitrate_per_stage,
    embedding_dim=embedding_dim,
    batch_size=args.batch,
    device=device,
)
ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt["end_to_end_model"])
model.to(device)
model.eval()

print(f"[Info] Loaded checkpoint from {args.ckpt}")

# # dataloader: 여기서 다른 데이터셋으로 바꾸고 싶으면 이 부분만 수정하면 됨
# imagenet_loader = ImageNet_Loader(args.batch, args.img_size, norm=args.norm)
# test_loader = imagenet_loader.dataloader_1k()["val"]

loader = Kodak_Patch_Loader(patch_size=(args.img_size, args.img_size), norm=args.norm)
test_loader = loader.dataloader(batch_size=args.batch)

# 역정규화 (train 코드와 동일)
mean, std = loader.norm  # -> list[list]
inv_std = [1 / s for s in std]
inv_mean = [-m / s for m, s in zip(mean, std)]
inv_norm = torchvision.transforms.Normalize(inv_mean, inv_std)

# modulation index sequence
m_idx = [int(d) for d in args.m]

os.makedirs(args.out_dir, exist_ok=True)


# -------------------- Metric helpers --------------------
def compute_psnr(x, y, data_range=1.0, eps=1e-8):
    """
    x, y: [B, C, H, W], 값 범위 [0,1] 기준 (data_range=1.0)
    """
    mse = F.mse_loss(x, y, reduction="none")
    mse = mse.reshape(mse.size(0), -1).mean(dim=1)  # per-image
    psnr = 10 * torch.log10(data_range ** 2 / (mse + eps))
    return psnr  # [B]


def tensor_to_lpips_range(x):
    """
    [0,1] -> [-1,1] 로 변환 (LPIPS 입력용)
    """
    return x * 2 - 1


# -------------------- 샘플 이미지 저장 함수 --------------------
def save_sample_grid(orig, recon, save_path, nrow=8):
    """
    orig, recon: [B, C, H, W] (이미 inv_norm, clamp된 [0,1] 라고 가정)
    """
    # 앞에서 B개만 사용 (B>=nrow*2 이상이면 적당히)
    B = min(orig.size(0), nrow)
    orig = orig[:B]
    recon = recon[:B]

    # [2B, C, H, W]로 concat (위: orig, 아래: recon)
    stacked = torch.cat([orig, recon], dim=0)
    grid = make_grid(stacked, nrow=nrow, padding=2, normalize=False)
    save_image(grid, save_path)
    print(f"[Save] {save_path}")


# -------------------- Test loop --------------------
results = []  # (channel, stage, snr)별 metric summary 저장용

for ch_name, apply_fading, equalizer in CHANNEL_MODES:
    print(f"\n========== Channel: {ch_name} (apply_fading={apply_fading}, eq={equalizer}) ==========")
    ch_dir = os.path.join(args.out_dir, ch_name)
    os.makedirs(ch_dir, exist_ok=True)

    for stage in STAGES:
        print(f"\n--- Stage {stage} ---")
        for snr_db in SNR_LIST:
            print(f"[Test] stage={stage}, SNR={snr_db} dB")

            psnr_vals = []
            msssim_vals = []
            lpips_vals = []

            # 샘플 이미지 저장 여부 (stage, snr, channel 조합당 한 번만)
            saved_sample = False

            test_pbar = tqdm(
                enumerate(test_loader, 1),
                total=len(test_loader),
                leave=False,
                desc=f"[{ch_name}] Stage {stage}, SNR {snr_db} dB",
            )
            for b_idx, (x, _) in test_pbar:
                x = x.to(device)  # [B, C, H, W]
                B = x.size(0)

                # SNR 벡터
                snr_vec = torch.full((B,), float(snr_db), device=device, dtype=torch.float32)

                with torch.no_grad():
                    rec, *_ = model(
                        x,
                        stage,
                        snr_vec,
                        m_idx,
                        nsvq=True,
                        rvq_activate=True,
                        apply_fading=apply_fading,
                        equalizer=equalizer,
                    )

                # --- 역정규화 후 [0,1]에 clamp ---
                x_denorm = inv_norm(x).clamp(0, 1)
                rec_denorm = inv_norm(rec).clamp(0, 1)

                # --- PSNR ---
                psnr_batch = compute_psnr(rec_denorm, x_denorm, data_range=1.0)  # [B]
                psnr_vals.extend(psnr_batch.cpu().tolist())

                # --- MS-SSIM ---
                msssim_batch = ms_ssim(
                    rec_denorm, x_denorm,
                    data_range=1.0,
                    size_average=False,
                    win_size=7,
                )  # [B]
                msssim_vals.extend(msssim_batch.cpu().tolist())

                # --- LPIPS ---
                x_lp = tensor_to_lpips_range(x_denorm)
                rec_lp = tensor_to_lpips_range(rec_denorm)
                lpips_batch = lpips_fn(rec_lp, x_lp)  # [B,1,1,1] 또는 [B]
                lpips_vals.extend(lpips_batch.view(-1).detach().cpu().tolist())

                # --- 샘플 이미지 저장 (특정 SNR에서만, 첫 배치만) ---
                if (not saved_sample) and (snr_db in SAVE_SNR_LIST):
                    sample_path = os.path.join(
                        ch_dir,
                        f"sample_stage{stage}_snr{snr_db}.png"
                    )
                    save_sample_grid(x_denorm, rec_denorm, sample_path, nrow=min(8, B))
                    saved_sample = True

            # 한 (channel, stage, snr) 조합에 대한 통계
            psnr_vals = np.array(psnr_vals)
            msssim_vals = np.array(msssim_vals)
            lpips_vals = np.array(lpips_vals)

            results.append({
                "channel": ch_name,
                "stage": stage,
                "snr_db": snr_db,
                "psnr_mean": float(psnr_vals.mean()),
                "psnr_std": float(psnr_vals.std(ddof=1)) if psnr_vals.size > 1 else 0.0,
                "msssim_mean": float(msssim_vals.mean()),
                "msssim_std": float(msssim_vals.std(ddof=1)) if msssim_vals.size > 1 else 0.0,
                "lpips_mean": float(lpips_vals.mean()),
                "lpips_std": float(lpips_vals.std(ddof=1)) if lpips_vals.size > 1 else 0.0,
                "num_samples": int(psnr_vals.size),
            })

# -------------------- 결과 저장 --------------------
df = pd.DataFrame(results)
csv_path = os.path.join(args.out_dir, f"metrics_{args.model}_L{args.stages}_{args.bits}b.csv")
df.to_csv(csv_path, index=False)
print(f"\n[Done] Saved metrics to {csv_path}")
print(df.head())
