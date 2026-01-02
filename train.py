import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.utils import make_grid, save_image
import numpy as np
import os
from dataloader import ImageNet_Loader
from networks import Digital_SemCom, Analog_SemCom, MobileNet_SemCom, MobileViT_SemCom
import argparse
from tqdm.auto import tqdm
import math
import logging
from pytorch_msssim import ms_ssim


logging.getLogger("PIL.PngImagePlugin").disabled = True
logging.getLogger("PIL.JpegImagePlugin").disabled = True
logging.getLogger("PIL.TiffImagePlugin").disabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDNN_CONV_USE_CUDNN_FRONTEND"] = "0"
torch.backends.cudnn.enabled = True  # 기본적으로 활성화
torch.backends.cudnn.benchmark = True  # 실행 계획 최적화 활성화
torch.backends.cudnn.deterministic = False  # 다양한 실행 계획을 시도하도록 설정


def log_and_replace(log_state, *, model, used_codebook_indices,
                    total_vq_loss, used_codebook_indices_list,
                    perplexity_list, args, max_stage, num_batch):

    if log_state["accum_steps"] == 0:
        return  # 누적된 게 없으면 패스

    acc_steps = log_state["accum_steps"]

    vq_loss_avg = log_state["vq_loss_acc"] / acc_steps
    commit_avg  = log_state["commit_acc"]  / acc_steps
    kl_avg      = log_state["kl_acc"]      / acc_steps
    perplex_avg = log_state["perplex_acc"] / acc_steps
    stage_prob  = log_state["stg_cnt"]     / acc_steps

    # 로그 리스트에 저장
    total_vq_loss.append(round(vq_loss_avg, 6))
    used_codebook_indices_list.append(used_codebook_indices)
    perplexity_list.append(perplex_avg)

    # 콘솔 출력
    print("\n")
    print(
        "[{}/{}] Training iter:{}, Total loss:{:.6f}, "
        "Commit:{:.6f}, KLD:{:.6f}".format(
            log_state["epoch"], log_state["max_epoch"],
            num_batch + 1, vq_loss_avg,
            commit_avg, kl_avg
        )
    )
    print(f"perplexity each stages = {[f'{x:.4f}' for x in perplex_avg]}")
    scatter_vals = [
        f"{v:.4f}"
        for v in model.resume.codebooks.var(dim=1).mean(dim=-1).tolist()
    ]
    print(f"codebook scattering {scatter_vals}")
    print(f"Stage prob : {stage_prob}")

    # codebook replacement
    model.resume.replace_unused_codebooks(
        max_stage=max_stage,
        only_display=False,
    )

    # state 리셋
    log_state["vq_loss_acc"]   = 0.0
    log_state["commit_acc"]    = 0.0
    log_state["kl_acc"]        = 0.0
    log_state["perplex_acc"]   = np.zeros(args.stages)
    log_state["stg_cnt"]       = np.zeros(args.stages)
    log_state["accum_steps"]   = 0


# Hyper-parameters
batch_size = 144
snr_max = 15
model_ver = 1

parser = argparse.ArgumentParser(description="Train UEP-RVQ")
parser.add_argument("--model", type=str, help="Model name", default='gauss')
parser.add_argument("--stages", type=int, help="The number of RVQ stages", default=4)
parser.add_argument("--bits", type=int, help="The number of RVQ codeword bits", default=12)
parser.add_argument("--batch", type=int, help="Batch size", default=144)
parser.add_argument("--version", type=str, help="Model version", default='1')
parser.add_argument("--dataset", type=str, help="Test dataset name", default='ImageNet')
parser.add_argument("--tune", type=bool, help="Fine-tunning", default=False)
parser.add_argument("--norm", action='store_true', help="Data Normalization")
parser.add_argument("--fading", action='store_true', help="Rayleigh fading channel")
parser.add_argument("--lr", type=float, help="Learning Rate", default=1e-4)
parser.add_argument("--m", type=str, help="Modulation index sequence", default='0000')
parser.add_argument("--kld_var", type=float, help="KLD sigma value", default=0.01)
parser.add_argument("--target_cbr", type=float, help="Target CBR for Analog Transmission", default=1/64)
parser.add_argument("--epochs", type=int, help="Number of training epochs", default=200)
parser.add_argument("--nsvq", dest="nsvq", action="store_true", help="Enable NSVQ (default: on)")
parser.add_argument("--no-nsvq", dest="nsvq", action="store_false", help="Disable NSVQ")
parser.set_defaults(nsvq=True)
args = parser.parse_args()

max_epoch = args.epochs
normal_mean = 0 #mean for normal distribution
normal_std = 1 #standard deviation for normal distribution
training_log_batches = 200 #number of batches to get logs of training
replacement_num_batches = 200 #number of batches to check codebook activity and discard inactive codebook vectors
best_loss = 2

SAVE_VAL_GRID = True
VAL_SAMPLE_IDX = 0          # (배치 내) 그리드에 쓸 참조 이미지 인덱스
GRID_NORM = True            # make_grid(normalize=True) 여부
GRID_PAD = 2

vq_bitrate_per_stage = args.bits
CR = 48 * 8 // vq_bitrate_per_stage
embedding_dim = 128
# embedding_dim = 96
channel = "awgn" if args.fading == False else "rayleigh"
ste_tag = "CANSVQ" if args.nsvq else "STE"
print("NSVQ =", args.nsvq)

if args.tune == False:
    save_title = f'{args.model}_{ste_tag}_{channel}_L{args.stages}_{args.bits}b_v{args.version}_{args.norm}_kld_{args.kld_var}'
else:
    args.lr = 1e-5
    save_title = f'{args.model}_{ste_tag}_{channel}_L{args.stages}_{args.bits}b_v{args.version}_{args.norm}_kld_{args.kld_var}_revised'

print(f'SAVE TITLE: {save_title}')

# Arrays to save the logs of training
total_vq_loss = [] # tracks VQ loss
used_codebook_indices_list = [] # tracks indices of used codebook entries
perplexity_list = [] # tracks perplexity
eval_loss = []
eval_ssim = []

if args.model == 'gauss':
    model = Digital_SemCom(num_hiddens=128,
                        num_residual_hiddens=64,
                        num_residual_layers=4,
                        num_stages=args.stages,
                        vq_bitrate_per_stage=vq_bitrate_per_stage,
                        embedding_dim=embedding_dim,
                        batch_size=args.batch,
                        device=device,
                        )
elif args.model == 'analog':
    model = Analog_SemCom(num_hiddens=128,
                        num_residual_hiddens=64,
                        num_residual_layers=4,
                        embedding_dim=embedding_dim,
                        target_CBR=args.target_cbr,
                        )
    
elif args.model == 'mobilenet':
    model = MobileNet_SemCom(num_stages=4, vq_bitrate_per_stage=12, embedding_dim=embedding_dim, batch_size=args.batch, device=device)
elif args.model == 'mobilevit':
    model = MobileViT_SemCom(num_stages=4, vq_bitrate_per_stage=12, embedding_dim=embedding_dim, batch_size=args.batch, device=device)

if args.tune == True:
    model_name = f'./output/{args.model}_{args.stages}stages_B{args.bits}_ver{args.version}_{args.norm}_kld_{args.kld_var}.pt'
    model.load_state_dict(torch.load(model_name)['end_to_end_model'])
    max_epoch = 100

model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
# ─────────────────────── Scheduler ───────────────────────
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer,
                              T_max=max_epoch,  # 전체 에폭 수
                              eta_min=1e-6)     # 최소 lr
# ────────────────────────────────────────────────────────────

print(f'Data norm: {args.norm}')
imagenet_loader = ImageNet_Loader(args.batch, 128, norm=args.norm)
train_loader = imagenet_loader.dataloader_1k()['train']
val_loader = imagenet_loader.dataloader_1k()['val']

# ------------------------------------------------------------------
# 0) 역-정규화 helper  (loader.norm 은 [mean, std])
# ------------------------------------------------------------------
mean, std = imagenet_loader.norm  # -> list[list]
inv_std  = [1 / s for s in std]
inv_mean = [-m / s for m, s in zip(mean, std)]
inv_norm = torchvision.transforms.Normalize(inv_mean, inv_std)



for epoch in range(max_epoch):
    epoch_loss_accum = 0.0
    log_state = {
        "epoch": epoch,
        "max_epoch": max_epoch,
        "vq_loss_acc": 0.0,
        "commit_acc": 0.0,
        "kl_acc": 0.0,
        "perplex_acc": np.zeros(args.stages),
        "stg_cnt": np.zeros(args.stages),
        "accum_steps": 0,
    }
    model.train()

    total_batches = len(train_loader)               # 한 epoch의 배치 수
    seg_len = math.ceil(total_batches / args.stages)      # 한 단계가 차지할 배치 길이

    # 예: [1 1 … 1  2 2 … 2  3 … ] (길이 = total_batches)
    stage_seq = (
        np.repeat(np.arange(1, args.stages + 1), seg_len)[:total_batches]
    )
    prev_stage = int(stage_seq[0])

    # ── tqdm wrapper ──────────────────────────────────────────────────────────
    train_pbar = tqdm(
        enumerate(train_loader, 1),
        total=len(train_loader),
        desc=f"[L{args.stages}B{vq_bitrate_per_stage}] Epoch {epoch+1}/{max_epoch}",
        leave=False,
        ncols=120,    
        unit="batch"
    )
    for num_batch, (data, label) in train_pbar:
        log_state["accum_steps"] += 1
        data = data.to(device)
        # sum_stages = np.random.randint(1, num_stages+1)
        sum_stages = int(stage_seq[num_batch - 1])      # 1 ··· num_stages
        log_state["stg_cnt"][0:sum_stages] += 1

        # snr_db = np.random.randint(0, snr_max+1)
        snr_scalar = np.random.uniform(0, snr_max)
        snr_db = torch.full((batch_size,), snr_scalar, device=device, dtype=torch.float32)

        # custom m_idx
        m_idx = [int(digit) for digit in args.m]
        # m_idx = sorted(np.random.choice(5, 4, replace=True))
        optimizer.zero_grad()

        rec_data, used_codebook_indices, perplexity, z_e, z_q, loss_commit = model(data, sum_stages, snr_db, m_idx, nsvq=args.nsvq, rvq_activate = True, apply_fading=args.fading, equalizer='zf')
        commit_loss = (z_e - z_q.detach()).square().mean()
        embedding_loss = (z_q - z_e.detach()).square().mean()
        var_loss = F.mse_loss(z_e, torch.zeros_like(z_e))

        target_variance = args.kld_var
        target_std = torch.sqrt(torch.tensor(target_variance))
        log_var = torch.log2(torch.var(z_q, dim=0) + 1e-9) # z_e: [~, 128]
        mu = torch.mean(z_q, dim=0)
        kl_divergence = -0.5 * torch.mean(1 + log_var - (mu / target_std).pow(2) - (log_var.exp() / target_variance))
        

        # vq_loss = F.mse_loss(rec_data, data) + 0.01*commit_loss + 0.1*(embedding_loss) +  0.1*var_loss + 0.01*kl_divergence # ver1
        vq_loss = F.mse_loss(rec_data, data) + 0.1*loss_commit +  0.1*var_loss + 0.01*kl_divergence # ver1-2
        # vq_loss = F.mse_loss(rec_data, data) + 0.01*commit_loss + 0.1*(embedding_loss) +  0.1*var_loss # ver2
        # vq_loss = F.mse_loss(rec_data, data) + 0.01*commit_loss + 0.1*(embedding_loss) # ver3
        # vq_loss = F.mse_loss(rec_data, data)
        # vq_loss =  F.mse_loss(rec_data, data) + 0.1*loss_commit

        vq_loss.backward()
        optimizer.step()

        # if args.model == 'gauss':
        #     while len(perplexity) < args.stages:
        #         perplexity.append(0.0)


        # epoch_loss_accum += vq_loss.item()

        # log_state["vq_loss_acc"] += vq_loss.item()
        # log_state["commit_acc"]  += commit_loss.item()
        # log_state["kl_acc"]      += kl_divergence.item()
        # if args.model == 'gauss':
        #     log_state["perplex_acc"] += np.array(perplexity)

        # 손실 값들 한 번만 .item() 호출해서 가져오기
        vq_loss_val      = float(vq_loss.item())
        commit_loss_val  = float(commit_loss.item())
        kl_div_val       = float(kl_divergence.item())

        epoch_loss_accum           += vq_loss_val
        log_state["vq_loss_acc"]   += vq_loss_val
        log_state["commit_acc"]    += commit_loss_val
        log_state["kl_acc"]        += kl_div_val

        if args.model != "analog":
            # perplexity 길이를 stages에 맞춰 패딩/슬라이스한 배열 생성
            perplex_arr = np.zeros(args.stages, dtype=np.float32)
            n = min(len(perplexity), args.stages)
            perplex_arr[:n] = np.array(perplexity[:n], dtype=np.float32)

            log_state["perplex_acc"] += perplex_arr

        # ── tqdm postfix 갱신 ────────────────────────────────────────────
        train_pbar.set_postfix(
            loss       = f"{vq_loss.item():.4f}",
            avg_loss   = f"{epoch_loss_accum / num_batch:.4f}",
            snr        = snr_db[0].item(),
            stage      = sum_stages,
        )

        if args.model != "analog" and num_batch < total_batches:
            next_stage = int(stage_seq[num_batch])  # 다음 배치의 stage

            # ── stage 변경 직전: 로그 + codebook replacement ──────────────
            if next_stage != sum_stages:
                # 1) 로그 출력
                log_and_replace(
                    log_state,
                    model=model,
                    used_codebook_indices=used_codebook_indices,
                    total_vq_loss=total_vq_loss,
                    used_codebook_indices_list=used_codebook_indices_list,
                    perplexity_list=perplexity_list,
                    args=args,
                    max_stage=sum_stages,  # 현재 stage까지
                    num_batch=num_batch,
                )

    if args.model != "analog" and log_state["accum_steps"] > 0:
        log_and_replace(
            log_state,
            model=model,
            used_codebook_indices=used_codebook_indices,
            total_vq_loss=total_vq_loss,
            used_codebook_indices_list=used_codebook_indices_list,
            perplexity_list=perplexity_list,
            args=args,
            max_stage=args.stages,   # 마지막에는 1~L 전부 replacement
            num_batch=num_batch,
        )

    train_pbar.close()

    print(f"[Epoch {epoch+1}] Avg train loss = {epoch_loss_accum / total_batches:.6f}")
    print("\nTraining Finished >>> Logs and Checkpoints Saved!!!")
    
    
    ####################### Validation #############################
    val_vq_loss_accumulator = 0.0
    val_ssim_accumulator   = 0.0
    save_address = './output/'
    VAL_GRID_DIR = f"{save_address}/validation/{args.nsvq}/{args.model}_{channel}_{args.stages}_{args.bits}/kld_{args.kld_var}"
    os.makedirs(VAL_GRID_DIR, exist_ok=True)
    DATA_RANGE = 2.0 if args.norm else 1.0
    model.eval()
    with torch.no_grad():
        # 첫 배치에서 ref_img 1장만 뽑음
        ref_img = None
        for val_num_batch, (val_data, _) in enumerate(val_loader):
            val_data = val_data.to(device)

            # -- 레이아웃용 첫 이미지 확보 -----------------------------
            if ref_img is None:
                ref_img = val_data[VAL_SAMPLE_IDX:VAL_SAMPLE_IDX + 1]   # [1,C,H,W]

            # (1) 성능 누적
            val_sum_stages = val_num_batch % args.stages + 1
            val_snr_db     = val_num_batch % (snr_max + 1)
            val_snr_db = torch.full((batch_size,), val_snr_db, device=device, dtype=torch.float32)
            val_rec, *_    = model(val_data, val_sum_stages, val_snr_db, m_idx, nsvq=args.nsvq, rvq_activate=True, apply_fading=args.fading, equalizer='zf')
            val_vq_loss_accumulator += F.mse_loss(val_rec, val_data).item()
            val_ssim_accumulator   += ms_ssim(val_rec, val_data, data_range=DATA_RANGE, size_average=True, win_size=7).item()

            # ----------------------------------------------------------
            # 격자 이미지는 epoch 당 한 번만 저장 → 첫 배치에서만 생성
            # ----------------------------------------------------------
            if val_num_batch == 0 and SAVE_VAL_GRID:
                grid_imgs = []

                # Row 0: original × (snr_max+1)  -----------------------------------
                for _ in range(snr_max + 1):
                    grid_imgs.append(inv_norm(ref_img[0].cpu()).clamp(0, 1))

                # Row 1-L: stage × snr --------------------------------------------
                for stage in range(1, args.stages + 1):
                    for snr in range(0, snr_max + 1):
                        snr = torch.full((batch_size,), snr, device=device, dtype=torch.float32)
                        out, *_ = model(ref_img, stage, snr, m_idx, nsvq=args.nsvq, rvq_activate=True, apply_fading=args.fading, equalizer='zf')
                        out_img = inv_norm(out[0].cpu()).clamp(0, 1)
                        grid_imgs.append(out_img)

                # (num_stages+1) 행, (snr_max+1) 열
                n_cols = snr_max + 1
                grid   = make_grid(torch.stack(grid_imgs), nrow=n_cols,
                                padding=GRID_PAD, normalize=False)

                png_path = f"{VAL_GRID_DIR}/{save_title}_epoch{epoch+1}.png"
                save_image(grid, png_path)
                print(f"[Val] saved image grid → {png_path}")

            

        # --------------------------- (3) 평균 통계 출력 ---------------------------
        avg_loss = val_vq_loss_accumulator / len(val_loader)
        avg_ssim = val_ssim_accumulator   / len(val_loader)
        print(f"Valid MSE Loss = {avg_loss:.6f} | MS-SSIM = {avg_ssim:.6f}\n")

        # checkpoint & 로그는 기존 코드 유지
        if avg_loss < best_loss:
            torch.save({"end_to_end_model": model.state_dict()},
                    f"{save_address}{save_title}.pt")
            best_loss = avg_loss

        eval_loss.append(round(avg_loss, 6))
        eval_ssim.append(round(avg_ssim, 6))
        np.save(f"{save_address}eval_loss_{save_title}.npy", np.asarray(eval_loss))
        np.save(f"{save_address}eval_ssim_{save_title}.npy", np.asarray(eval_ssim))

    scheduler.step()