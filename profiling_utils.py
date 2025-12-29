import time
import numpy as np
import torch
import torch.nn as nn


# -------------------------
# Params
# -------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# -------------------------
# FLOPs / MACs (Conv2d + Linear only)
# -------------------------
@torch.no_grad()
def estimate_macs_flops_conv_linear(
    model: nn.Module,
    example_inputs: tuple,
    forward_kwargs: dict,
    assume_mul_add_as_2flops: bool = True,
) -> dict:
    """
    Returns dict with:
      - macs_conv_linear
      - flops_conv_linear  (if assume_mul_add_as_2flops)
    This only counts Conv2d + Linear via forward hooks.
    """

    macs = {"total": 0}

    def conv_hook(module: nn.Conv2d, inp, out):
        # inp[0]: [B, Cin, Hin, Win]
        x = inp[0]
        if not isinstance(out, torch.Tensor):
            return
        B = out.shape[0]
        Cout = out.shape[1]
        Hout = out.shape[2]
        Wout = out.shape[3]
        Cin = module.in_channels
        kH, kW = module.kernel_size
        groups = module.groups
        # MACs = B * Hout * Wout * Cout * (Cin/groups * kH * kW)
        mac = B * Hout * Wout * Cout * (Cin // groups) * kH * kW
        macs["total"] += int(mac)

    def linear_hook(module: nn.Linear, inp, out):
        x = inp[0]
        if not isinstance(out, torch.Tensor):
            return
        # x: [..., in_features], out: [..., out_features]
        in_f = module.in_features
        out_f = module.out_features
        # number of "rows" = total elements / feature_dim
        rows = x.numel() // in_f
        mac = rows * in_f * out_f
        macs["total"] += int(mac)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    _ = model(*example_inputs, **forward_kwargs)

    for h in hooks:
        h.remove()

    mac_total = macs["total"]
    if assume_mul_add_as_2flops:
        return {"macs_conv_linear": mac_total, "flops_conv_linear": 2 * mac_total}
    return {"macs_conv_linear": mac_total}


@torch.no_grad()
def estimate_flops_with_torch_profiler(
    model: nn.Module,
    example_inputs: tuple,
    forward_kwargs: dict,
    warmup: int = 5,
    iters: int = 10,
) -> dict:
    """
    Estimate total FLOPs using torch.profiler(with_flops=True).
    This can include attention/matmul if the torch build reports it.
    Returns:
      - flops_profiler_total (int or None)
    """
    try:
        import torch.profiler as prof
    except Exception:
        return {"flops_profiler_total": None}

    model.eval()
    device = example_inputs[0].device
    activities = [prof.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(prof.ProfilerActivity.CUDA)

    # warmup
    for _ in range(warmup):
        _ = model(*example_inputs, **forward_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    with prof.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_flops=True,
    ) as p:
        for _ in range(iters):
            _ = model(*example_inputs, **forward_kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()

    total = 0
    any_flops = False
    for evt in p.key_averages():
        fl = getattr(evt, "flops", None)
        if fl is not None and fl > 0:
            total += int(fl)
            any_flops = True

    return {"flops_profiler_total": int(total) if any_flops else None}

# -------------------------
# Latency + Peak Memory
# -------------------------
@torch.no_grad()
def benchmark_latency_and_memory(
    model: nn.Module,
    example_inputs: tuple,
    forward_kwargs: dict,
    warmup: int = 30,
    iters: int = 100,
) -> dict:
    model.eval()

    device = example_inputs[0].device
    is_cuda = (device.type == "cuda")

    # Warmup
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        _ = model(*example_inputs, **forward_kwargs)
    if is_cuda:
        torch.cuda.synchronize()

    # Timed runs
    times_ms = []

    if is_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(iters):
            starter.record()
            _ = model(*example_inputs, **forward_kwargs)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))

        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    else:
        # CPU timing
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(*example_inputs, **forward_kwargs)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        peak_alloc = None
        peak_reserved = None

    times_ms = np.array(times_ms, dtype=np.float64)
    out = {
        "lat_mean_ms": float(times_ms.mean()),
        "lat_p50_ms": float(np.percentile(times_ms, 50)),
        "lat_p95_ms": float(np.percentile(times_ms, 95)),
        "peak_mem_alloc_bytes": int(peak_alloc) if peak_alloc is not None else None,
        "peak_mem_reserved_bytes": int(peak_reserved) if peak_reserved is not None else None,
    }
    return out


def bytes_to_mib(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


# -------------------------
# One-shot model profile
# -------------------------
@torch.no_grad()
def profile_one_model(
    name: str,
    model: nn.Module,
    example_inputs: tuple,
    forward_kwargs: dict,
) -> dict:
    params = count_parameters(model)

    # fast / stable: Conv2d + Linear only
    macs_flops = estimate_macs_flops_conv_linear(model, example_inputs, forward_kwargs)

    # slower / more complete: profiler FLOPs (may still be partial)
    prof_flops = estimate_flops_with_torch_profiler(model, example_inputs, forward_kwargs)

    bench = benchmark_latency_and_memory(model, example_inputs, forward_kwargs)

    out = {
        "name": name,
        "params": params,
        **macs_flops,
        **prof_flops,
        **bench,
    }
    return out

def pretty_print_profiles(profiles: list[dict]):
    header = (
        f"{'Model':18s} | {'Params(M)':>9s} | {'FLOPs(G) conv+lin':>16s} | "
        f"{'FLOPs(G) profiler':>16s} | {'Lat(ms)':>8s} | {'p50':>8s} | {'p95':>8s} | {'PeakMem(MiB)':>12s}"
    )
    print(header)
    print("-" * len(header))

    for p in profiles:
        params_m = p["params"] / 1e6

        flops_conv = p.get("flops_conv_linear", None)
        flops_conv_g = (flops_conv / 1e9) if (flops_conv is not None) else float("nan")

        flops_prof = p.get("flops_profiler_total", None)
        flops_prof_g = (flops_prof / 1e9) if (flops_prof is not None) else float("nan")

        lat = p["lat_mean_ms"]
        p50 = p["lat_p50_ms"]
        p95 = p["lat_p95_ms"]

        peak = p["peak_mem_alloc_bytes"]
        peak_mib = bytes_to_mib(peak) if peak is not None else float("nan")

        print(
            f"{p['name'][:18]:18s} | {params_m:9.3f} | {flops_conv_g:16.3f} | "
            f"{flops_prof_g:16.3f} | {lat:8.3f} | {p50:8.3f} | {p95:8.3f} | {peak_mib:12.2f}"
        )

    print("\nNotes:")
    print("- FLOPs(G) conv+lin: forward-hook count of Conv2d + Linear only (fast, stable).")
    print("- FLOPs(G) profiler: torch.profiler(with_flops=True) total (more complete; may be None if unsupported).")

def build_fwd_kwargs(device, batch_size: int):
    # snr를 batch 길이로 맞춰주는 게 대부분의 구현에서 안전함
    return dict(
        sum_stage=4,
        snr=torch.full((batch_size,), 10.0, device=device),
        m_idx=[0, 1, 2, 3],
        apply_fading=False,
        rvq_activate=True,
        nsvq=True,
        equalizer="zf",
    )

@torch.no_grad()
def try_batch_under_mem(builder_fn, batch_size: int, device, max_mem_mib: float,
                        H=128, W=128, warmup=2, use_reserved=True):
    """
    Returns: (ok: bool, peak_mib: float, err: str|None)
    - use_reserved=True: torch.cuda.max_memory_reserved 기준(더 보수적)
    - use_reserved=False: torch.cuda.max_memory_allocated 기준
    """
    if device.type != "cuda":
        # CPU면 메모리 threshold 의미가 없으니 항상 ok 취급
        return True, 0.0, None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    try:
        model = builder_fn(batch_size).to(device).eval()
        x = torch.randn(batch_size, 3, H, W, device=device)
        fwd_kwargs = build_fwd_kwargs(device, batch_size)

        # warmup forward (피크 메모리는 여기서 잡히는 경우가 많음)
        for _ in range(warmup):
            _ = model(x, **fwd_kwargs)

        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_reserved(device) if use_reserved else torch.cuda.max_memory_allocated(device)
        peak_mib = peak_bytes / (1024 ** 2)

        ok = (peak_mib <= max_mem_mib)
        return ok, peak_mib, None

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            torch.cuda.empty_cache()
            return False, float("inf"), "OOM"
        return False, float("inf"), repr(e)

def find_max_batch(builder_fn, device, max_mem_mib: float,
                   start_bs=1, max_bs_cap=1024, use_reserved=True):
    """
    Exponential grow -> binary search for maximum batch size under memory threshold.
    Returns: (best_bs, best_peak_mib)
    """
    # Check bs=1 first
    ok, peak, err = try_batch_under_mem(builder_fn, start_bs, device, max_mem_mib, use_reserved=use_reserved)
    if not ok:
        return 0, peak  # even bs=1 fails

    best_bs, best_peak = start_bs, peak

    # Exponential search
    bs = start_bs
    while True:
        nxt = bs * 2
        if nxt > max_bs_cap:
            break
        ok, peak, err = try_batch_under_mem(builder_fn, nxt, device, max_mem_mib, use_reserved=use_reserved)
        if ok:
            best_bs, best_peak = nxt, peak
            bs = nxt
        else:
            hi = nxt
            lo = bs
            break
    else:
        hi = bs

    # If we never broke due to failure, we might be at cap
    if best_bs == max_bs_cap:
        return best_bs, best_peak

    # If we didn't define hi via failure, set hi = best_bs + 1 for binary loop to noop
    if "hi" not in locals():
        return best_bs, best_peak

    # Binary search between (lo, hi)
    left, right = lo, hi - 1
    while left <= right:
        mid = (left + right) // 2
        ok, peak, err = try_batch_under_mem(builder_fn, mid, device, max_mem_mib, use_reserved=use_reserved)
        if ok:
            best_bs, best_peak = mid, peak
            left = mid + 1
        else:
            right = mid - 1

    return best_bs, best_peak
