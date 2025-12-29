import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import uniform, normal

def generate_qam_codebook(M, P=1):
    if M == 1:
        # BPSK case: {-1, 1}
        codebook = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    elif np.log2(M) % 2 == 0:
        m = int(torch.sqrt(torch.tensor(M, dtype=torch.float32)).item())

        real_part = torch.arange(-m + 1, m, 2, dtype=torch.float32)
        imag_part = torch.arange(-m + 1, m, 2, dtype=torch.float32)

        codebook = torch.complex(
            real_part.repeat(m), 
            imag_part.repeat_interleave(m)
        )
    else:
        raise ValueError('Invalid M')

    avg_power = (codebook.abs() ** 2).mean()
    codebook = codebook / torch.sqrt(avg_power / P)

    return codebook


class ResUME(nn.Module):
    """
    Parameters
    ----------
    num_stages : int
        L – number of residual stages.
    vq_bitrate_per_stage : int
        b – bits per vector per stage (K = 2**b codewords).
    data_dim : int
        Vector dimension D.
    batch_size : float
        Used only for discard_threshold heuristic.
    device : torch.device
        Target device.
    """

    def __init__(
        self,
        num_stages: int,
        vq_bitrate_per_stage: int,
        data_dim: int,
        batch_size: float,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()

        # ------------------- basic configs ----------------------------------
        self.num_stages = num_stages
        self.K          = 1 << vq_bitrate_per_stage
        self.D          = data_dim
        self.device     = device
        self.dtype      = torch.float32

        # ------------------- symbol codebooks (QAM) -------------------------
        self.symbol_codebook = [
            generate_qam_codebook(2 ** (2 * m)).to(device) for m in range(5)
        ]

        # ------------------- VQ codebooks -----------------------------------
        init = uniform.Uniform(-1, 1).sample([num_stages, self.K, self.D]).to(device)
        self.codebooks = nn.Parameter(init, requires_grad=True)
        self.register_buffer("codebooks_used", torch.zeros(num_stages, self.K, dtype=torch.int32))

        self.discard_threshold = 64 * 0.05 * batch_size / self.K
        self.eps = 1e-8
        self.beta = 0.01
        self.normal_dist = normal.Normal(0, 1)
        self.register_buffer("stage_batches", torch.zeros(num_stages, dtype=torch.int32))

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x, sum_stages, snr_db, m_idx, apply_fading, nsvq: bool = True , equalizer='zf'):
        if sum_stages > self.num_stages:
            raise ValueError("sum_stages exceeds available stages")

        residual  = x
        quantized_list, remainder_list = [], []
        indices_list, rx_vecs, perplexities = [], [], []
        loss_codebook = 0.0
        loss_commitment = 0.0

        for l in range(0, len(m_idx)):
            cb = self.codebooks[l]

            # ----------- (1) residue setting -----------------
            residual_d = residual

            # ----------- (2) hard VQ -------------------------------------
            q, r, idx = self.hard_vq(residual_d, cb)

            # ----------- (3) channel mapping -----------------------------
            n_bits   = 2 * m_idx[l]
            packed   = self.convert_mbit_to_nbit_tensor(
                idx, n_bits, input_bit_width=int(np.log2(self.K))
            )
            rx_bits  = self.channel(
                packed, snr_db, m_idx[l],
                apply_fading=apply_fading,
                equalizer=equalizer,
            )
            rx_idx   = self.convert_nbit_to_mbit_tensor(
                rx_bits, n_bits, output_bit_width=int(np.log2(self.K))
            )
            rx_vec   = cb[rx_idx]

            # ------------(4) loss update ---------------------------------
            loss_codebook += F.mse_loss(q, residual.detach(), reduction='mean')
            loss_commitment += self.beta * F.mse_loss(residual, q.detach(), reduction='mean')

            # ----------- (5) bookkeeping ---------------------------------
            quantized_list.append(q)
            remainder_list.append(r)
            indices_list.append(idx)
            rx_vecs.append(rx_vec)
            perplexities.append(self.calculate_perplexity(idx))

            residual = r  # pass residual to next stage

            self.stage_batches[l] += 1

        recon = sum(rx_vecs[:sum_stages])
        loss = loss_codebook + loss_commitment

        # usage stats
        with torch.no_grad():
            for l, idx in enumerate(indices_list):
                self.codebooks_used[l] += torch.bincount(idx, minlength=self.K)

        if self.training:
            if nsvq:
                out = self.noise_substitution_vq(x, recon)
            else:
                # straight-through style
                out = (recon - x).detach() + x
            return out, self.codebooks_used.cpu().numpy(), perplexities, loss
        else:
            return recon.detach(), self.codebooks_used.cpu().numpy(), perplexities, loss

    # ───────────────────────── helpers ───────────────────────────────────
    def hard_vq(self, x: torch.Tensor, codebook: torch.Tensor):
        dists = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ codebook.t()
            + codebook.pow(2).sum(1).unsqueeze(0)
        )
        idx = dists.argmin(1)
        q   = codebook[idx]
        residual = x - q
        return q, residual, idx

    def noise_substitution_vq(self, input_data, hard_quantized_input):
        random_vector = self.normal_dist.sample(input_data.shape).to(input_data.device)
        norm_hard_quantized_input = (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True)
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True)
        avg_mean = (hard_quantized_input - input_data).mean(dim=1, keepdim=True)
        vq_error = ((norm_hard_quantized_input / norm_random_vector).sqrt() * random_vector) + avg_mean
        return input_data + vq_error

    def channel(self, indices, snr_db, m_index, apply_fading=False, block_fading=True, equalizer='zf'):
        symbol = self.symbol_codebook[m_index][indices]  # shape: [...,]
        if apply_fading:
            if block_fading:
                h_shape = (1,) * symbol.ndim
            else:
                h_shape = symbol.shape
            real = torch.randn(h_shape, device=self.device) / math.sqrt(2)
            imag = torch.randn(h_shape, device=self.device) / math.sqrt(2)
            h = real + 1j * imag
            faded = symbol * h
        else:
            faded = symbol
            h = torch.ones_like(symbol)

        snr_linear    = 10 ** (snr_db[0] / 10)
        noise_power   = torch.mean(torch.abs(symbol) ** 2) / snr_linear
        noise_std_dev = torch.sqrt(noise_power / 2)
        noise = noise_std_dev * (
            torch.randn_like(faded) + 1j * torch.randn_like(faded)
        )
        rx = faded + noise
        if equalizer == 'zf':
            # Zero-Forcing
            y_eq = rx / h
            flat = y_eq.view(-1)
            cb   = self.symbol_codebook[m_index].unsqueeze(0)
            dist = torch.abs(flat.unsqueeze(1) - cb)
            decisions = dist.argmin(dim=1).view(indices.shape)

        elif equalizer == 'mmse':
            mmse_coeff = torch.conj(h) / (torch.abs(h)**2 + noise_power)
            y_mmse   = mmse_coeff * rx
            flat = y_mmse.view(-1)
            cb   = self.symbol_codebook[m_index].unsqueeze(0)
            dist = torch.abs(flat.unsqueeze(1) - cb)
            decisions = dist.argmin(dim=1).view(indices.shape)
        else:
            raise ValueError('Not implemented for other equalizer')

        return decisions

    # ----------------------------- util ---------------------------------
    def calculate_perplexity(self, idx: torch.Tensor) -> float:
        counts = torch.bincount(idx, minlength=self.K).float()
        prob   = counts / counts.sum()
        entropy = -(prob * (prob + 1e-10).log()).sum()
        return torch.exp(entropy).item()

    # bit-packing helpers remain identical
    def convert_mbit_to_nbit_tensor(self, data, n_bits, input_bit_width=8):
        masks = torch.tensor(
            [1 << (input_bit_width - 1 - i) for i in range(input_bit_width)],
            dtype=torch.int, device=data.device
        )
        bits  = (((data.unsqueeze(1) & masks) > 0).to(torch.uint8)).flatten()
        if n_bits == 0:
            return bits.to(torch.int)
        bits  = bits[: bits.numel() // n_bits * n_bits].view(-1, n_bits)
        powers = 2 ** torch.arange(n_bits - 1, -1, -1, dtype=torch.uint8, device=data.device)
        return (bits * powers).sum(1)

    def convert_nbit_to_mbit_tensor(self, data, n_bits, output_bit_width=8):
        if n_bits == 0:
            n_bits = 1
        bits = (((data.unsqueeze(1) & (1 << torch.arange(n_bits - 1, -1, -1, device=self.device))) > 0)
                 .to(torch.uint8)).flatten()
        bits = bits.view(-1, output_bit_width)
        powers = 2 ** torch.arange(output_bit_width - 1, -1, -1, dtype=torch.int, device=self.device)
        return (bits * powers).sum(1)

    # ------------------------- replacement ------------------------------
    def replace_unused_codebooks(self, max_stage=None, only_display=False):
        """
        max_stage:
        - None: 모든 stage (기존 동작 유지)
        - k: 1..k stage (index 0..k-1)만 replacement
        """
        with torch.no_grad():
            if max_stage is None:
                max_stage = self.num_stages

            max_stage = min(max_stage, self.num_stages)

            for l in range(max_stage):
                batches_l = max(self.stage_batches[l].item(), 1)
                usage = self.codebooks_used[l].float() / batches_l
                unused = (usage < self.discard_threshold).nonzero().squeeze(1)
                used   = (usage >= self.discard_threshold).nonzero().squeeze(1)

                if not only_display:
                    if used.numel() == 0:  # fallback
                        self.codebooks[l] += self.eps * torch.randn_like(self.codebooks[l])
                    else:
                        reps = int(np.ceil(unused.numel() / used.numel()))
                        new_vecs = (
                            self.codebooks[l, used]
                            .repeat(reps, 1)[torch.randperm(reps * used.numel())][: unused.numel()]
                        )
                        self.codebooks[l, unused] = new_vecs + self.eps * torch.randn_like(new_vecs)

                print(
                    f"[Replace] stage {l+1}: {unused.numel()} / {self.K} inactive "
                    f"(usage threshold {self.discard_threshold:.3e})"
                )

                # 이 stage에 대해서만 카운터 reset
                self.codebooks_used[l].zero_()
                self.stage_batches[l] = 0