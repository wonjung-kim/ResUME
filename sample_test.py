import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from networks import Digital_SemCom
from PIL import Image

# -------------------- Setting --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# -------------------- Sample Dataset --------------------
class SampleFolderDataset(Dataset):
    """
    sample_dir 안의 이미지(jpg, png, ...)들을 불러오는 간단한 Dataset
    """
    IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: str, img_size: int = 128, norm: bool = True):
        self.root = root
        self.paths = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.lower().endswith(self.IMG_EXT)
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No image files found in {root}")

        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        t_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # PIL -> Tensor [0,1]
        ]
        if norm:
            t_list.append(transforms.Normalize(self.mean, self.std))

        self.transform = transforms.Compose(t_list)
        self.norm = norm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")  # PIL Image
        img = self.transform(img)              # Tensor [C,H,W]
        name = os.path.splitext(os.path.basename(path))[0]
        return img, name


# -------------------- Metric helpers --------------------
def compute_psnr(x, y, data_range=1.0, eps=1e-8):
    mse = F.mse_loss(x, y, reduction="none")
    mse = mse.reshape(mse.size(0), -1).mean(dim=1)  # per-image
    psnr = 10 * torch.log10(data_range ** 2 / (mse + eps))
    return psnr  # [B]


# -------------------- main --------------------
def main():
    parser = argparse.ArgumentParser(description="Sample image test for Digital_SemCom")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to trained checkpoint (.pt)")
    parser.add_argument("--sample_dir", type=str, required=True,
                        help="Path to folder containing sample images")
    parser.add_argument("--m", type=str, default="0123",
                        help="Modulation index sequence (e.g., '0123')")
    parser.add_argument("--stages", type=int, default=4,
                        help="Number of RVQ stages (must match training)")
    parser.add_argument("--bits", type=int, default=12,
                        help="RVQ codeword bits (must match training)")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Image size (must match training/model)")
    parser.add_argument("--out_dir", type=str, default="./sample_test_out",
                        help="Directory to save outputs")
    parser.add_argument("--batch", type=int, default=3,
                        help="Batch size for testing (default=3)")
    parser.add_argument("--no_norm", action="store_true",
                        help="Do NOT apply normalization (use raw [0,1])")
    parser.add_argument("--snrs", type=str, default="0,5,10,15,20",
                        help="Comma-separated SNR list in dB (e.g., '0,5,10,15,20')")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------- Test Setting --------------------
    SNR_LIST = [
        float(s.strip()) for s in args.snrs.split(",") if s.strip() != ""
    ]
    print(f"[Info] Using SNR list (dB): {SNR_LIST}")

    # stage 1 ~ args.stages
    STAGES = list(range(1, args.stages + 1))

    CHANNEL_MODES = [
        ("awgn",     False, "zf"),  # AWGN
        ("rayleigh", True,  "zf"),  # Rayleigh + ZF
    ]

    # -------------------- 모델 로딩 --------------------
    vq_bitrate_per_stage = args.bits
    embedding_dim = 128  # 학습 때 사용한 값과 동일해야 함 (필요시 수정)

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

    # -------------------- dataloader: sample folder --------------------
    norm_flag = not args.no_norm
    dataset = SampleFolderDataset(
        root=args.sample_dir,
        img_size=args.img_size,
        norm=norm_flag,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if norm_flag:
        mean = dataset.mean
        std = dataset.std
        inv_std = [1.0 / s for s in std]
        inv_mean = [-m / s for m, s in zip(mean, std)]
        inv_norm = transforms.Normalize(inv_mean, inv_std)
    else:
        inv_norm = lambda x: x

    # modulation index sequence
    m_idx = [int(d) for d in args.m]

    # -------------------- Test loop --------------------
    for ch_name, apply_fading, equalizer in CHANNEL_MODES:
        print(f"\n========== Channel: {ch_name} (apply_fading={apply_fading}, eq={equalizer}) ==========")
        ch_dir = os.path.join(args.out_dir, ch_name)
        os.makedirs(ch_dir, exist_ok=True)

        for stage in STAGES:
            print(f"\n--- Stage {stage} ---")
            stage_dir = os.path.join(ch_dir, f"stage{stage}")
            os.makedirs(stage_dir, exist_ok=True)

            for snr_db in SNR_LIST:
                print(f"[Test] stage={stage}, SNR={snr_db} dB")
                snr_dir = os.path.join(stage_dir, f"snr{snr_db}")
                os.makedirs(snr_dir, exist_ok=True)

                psnr_vals = []

                test_pbar = tqdm(
                    enumerate(test_loader, 1),
                    total=len(test_loader),
                    leave=False,
                    desc=f"[{ch_name}] Stage {stage}, SNR {snr_db} dB",
                )

                for b_idx, (x, names) in test_pbar:
                    x = x.to(device)  # [B, C, H, W]
                    B = x.size(0)

                    snr_vec = torch.full(
                        (B,), float(snr_db),
                        device=device,
                        dtype=torch.float32
                    )

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

                    x_denorm = inv_norm(x).clamp(0, 1)
                    rec_denorm = inv_norm(rec).clamp(0, 1)

                    # PSNR
                    psnr_batch = compute_psnr(rec_denorm, x_denorm, data_range=1.0)
                    psnr_vals.extend(psnr_batch.cpu().tolist())

                    for j in range(B):
                        name = names[j]
                        out_img = rec_denorm[j]  # [C, H, W], [0,1]
                        out_path = os.path.join(
                            snr_dir,
                            f"{name}_ch-{ch_name}_L{stage}_snr{snr_db}.png"
                        )
                        save_image(out_img, out_path)

                psnr_vals = np.array(psnr_vals)
                psnr_mean = float(psnr_vals.mean())
                print(f"[Result] channel={ch_name}, stage={stage}, snr={snr_db} dB -> PSNR={psnr_mean:.2f} dB")

    print("\n[Done] All tests finished.")


if __name__ == "__main__":
    main()
