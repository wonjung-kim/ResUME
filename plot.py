#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 실험별 스타일 설정 (exp_name = csv가 들어있는 폴더 이름)
# ----------------------------------------------------
STYLE_CONFIG = {
    # 예시:
    "NQ": {
        "label": "NQ",
        "color": "black",
        "marker": "o",
        "linestyle": "-",
        "empty": True,  # True로 하면 비어있는 마커(마커 안이 흰색)
    },
    "VQ_16": {
        "label": "VQ (16)",
        "color": "C0",
        "marker": "o",
        "linestyle": "-",
        "empty": True,  # True로 하면 비어있는 마커(마커 안이 흰색)
    },
    "RVQ_2_8": {
        "label": "RVQ (2,8)",
        "color": "C1",
        "marker": "o",
        "linestyle": "-",
        "empty": True,  # True로 하면 비어있는 마커(마커 안이 흰색)
    },
    "RVQ_2_8_UME": {
        "label": "RVQ (2,8)+UME",
        "color": "C1",
        "marker": "o",
        "linestyle": "-",
        "empty": False,  # True로 하면 비어있는 마커(마커 안이 흰색)
    },
    "RVQ_4_4": {
        "label": "RVQ (4,4)",
        "color": "C2",
        "marker": "o",
        "linestyle": "-",
        "empty": True,  # True로 하면 비어있는 마커(마커 안이 흰색)
    },
    "RVQ_4_4_UME": {
        "label": "RVQ (4,4)+UME",
        "color": "C2",
        "marker": "o",
        "linestyle": "-",
        "empty": False,  # True로 하면 비어있는 마커(마커 안이 흰색)
    },
}

PLOT_ORDER = [
    # 예:
    "NQ",
    "VQ_16",
    "RVQ_2_8",
    "RVQ_2_8_UME",
    "RVQ_4_4",
    "RVQ_4_4_UME",
]

# STYLE_CONFIG = {
#     # 예시:
#     "KLD_0.001": {
#         "label": "sigma=0.001",
#         "color": "C0",
#         "marker": "o",
#         "linestyle": "-",
#         "empty": False,  # True로 하면 비어있는 마커(마커 안이 흰색)
#     },
#     "KLD_0.01": {
#         "label": "sigma=0.01",
#         "color": "C1",
#         "marker": "o",
#         "linestyle": "-",
#         "empty": False,  # True로 하면 비어있는 마커(마커 안이 흰색)
#     },
#     "KLD_0.1": {
#         "label": "sigma=0.1",
#         "color": "C2",
#         "marker": "o",
#         "linestyle": "-",
#         "empty": False,  # True로 하면 비어있는 마커(마커 안이 흰색)
#     },
#     "KLD_1.0": {
#         "label": "sigma=1.0",
#         "color": "C3",
#         "marker": "o",
#         "linestyle": "-",
#         "empty": False,  # True로 하면 비어있는 마커(마커 안이 흰색)
#     },
# }

# PLOT_ORDER = [
#     # 예:
#     "KLD_0.001",
#     "KLD_0.01",
#     "KLD_0.1",
#     "KLD_1.0",
# ]

def find_metric_csvs(root_dir: str) -> List[str]:
    csv_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".csv"):
                csv_paths.append(os.path.join(dirpath, fname))
    return sorted(csv_paths)


def load_experiments(csv_paths: List[str]):
    """
    각 csv를 읽어서 (exp_name, channel) -> DataFrame 로 정리.
    stage 컬럼이 있으면 마지막 stage만 사용.
    """
    experiments: Dict[Tuple[str, str], pd.DataFrame] = {}

    for path in csv_paths:
        df = pd.read_csv(path)
        if "snr_db" not in df.columns:
            continue

        if "stage" in df.columns:
            last_stage = df["stage"].max()
            df = df[df["stage"] == last_stage]

        exp_name = os.path.basename(os.path.dirname(path))

        if "channel" in df.columns:
            for ch, g in df.groupby("channel"):
                key = (exp_name, str(ch))
                experiments[key] = g.sort_values("snr_db")
        else:
            key = (exp_name, "unknown")
            experiments[key] = df.sort_values("snr_db")

    return experiments


def plot_metric_per_channel(
    experiments: Dict[Tuple[str, str], pd.DataFrame],
    metric: str,
    metric_pretty: str,
    out_dir: str,
    use_std: bool,
):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    })
    std_col = metric.replace("_mean", "_std")
    channels = sorted({ch for (_, ch) in experiments.keys()})

    for ch in channels:
        plt.figure()

        channel_keys = [(exp_name, ch_name) for (exp_name, ch_name) in experiments.keys() if ch_name == ch]

        ordered_keys: List[Tuple[str, str]] = []
        for name in PLOT_ORDER:
            for (exp_name, ch_name) in channel_keys:
                if exp_name == name and (exp_name, ch_name) not in ordered_keys:
                    ordered_keys.append((exp_name, ch_name))
        for (exp_name, ch_name) in sorted(channel_keys, key=lambda x: x[0]):
            if (exp_name, ch_name) not in ordered_keys:
                ordered_keys.append((exp_name, ch_name))

        for (exp_name, ch_name) in ordered_keys:
            df = experiments[(exp_name, ch_name)]
            snrs = df["snr_db"].values
            ys = df[metric].values

            style = STYLE_CONFIG.get(exp_name, {})
            label = style.get("label", exp_name)
            color = style.get("color", None)
            marker = style.get("marker", "o")
            linestyle = style.get("linestyle", "-")
            empty = style.get("empty", False)

            common_kwargs = {"label": label}
            if color is not None:
                common_kwargs["color"] = color
            if empty:
                common_kwargs["markerfacecolor"] = "none"

            if use_std and std_col in df.columns:
                yerr = df[std_col].values
                plt.errorbar(snrs, ys, yerr=yerr, marker=marker, linestyle=linestyle, capsize=3, **common_kwargs)
            else:
                plt.plot(snrs, ys, marker=marker, linestyle=linestyle, **common_kwargs)

        plt.xlabel("SNR (dB)")
        plt.ylabel(metric_pretty)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        os.makedirs(out_dir, exist_ok=True)
        fname = f"exp1_{metric}_{ch}.pdf"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()


# -------------------------------------------------------------------------
# NEW: improvement computation for *_UME vs naive
# -------------------------------------------------------------------------
LOWER_IS_BETTER = {"lpips_mean"}  # extend if needed


def _psnr_db_to_rel_mse(psnr_db: np.ndarray) -> np.ndarray:
    """
    PSNR(dB) = 10 log10(MAX^2 / MSE)
    Relative MSE (up to constant MAX^2) is proportional to 10^(-PSNR/10).
    Constant cancels in ratios, so this is safe for % comparisons.
    """
    return np.power(10.0, -psnr_db / 10.0)


def find_ume_pairs(experiments: Dict[Tuple[str, str], pd.DataFrame]) -> List[Tuple[str, str]]:
    """
    Find (base, ume) pairs by name: 'XXX_UME' vs 'XXX'.
    """
    names = sorted({exp_name for (exp_name, _) in experiments.keys()})
    name_set = set(names)

    pairs = []
    for n in names:
        if n.endswith("_UME"):
            base = n[:-4]
            if base in name_set:
                pairs.append((base, n))
    return sorted(pairs)


def compute_ume_improvements(
    experiments: Dict[Tuple[str, str], pd.DataFrame],
    pairs: List[Tuple[str, str]],
    metrics: List[str],
) -> pd.DataFrame:
    """
    Return a tidy DataFrame with per-(pair, channel, snr) improvement(%).

    Rules:
    - psnr_mean: convert to linear MSE scale (10^(-PSNR/10)), then compute MSE reduction(%)
    - lower-is-better metrics: reduction(%)
    - higher-is-better metrics: increase(%)
    """
    rows = []

    # channels present
    channels = sorted({ch for (_, ch) in experiments.keys()})

    for base_name, ume_name in pairs:
        for ch in channels:
            k_base = (base_name, ch)
            k_ume = (ume_name, ch)
            if k_base not in experiments or k_ume not in experiments:
                continue

            df_b = experiments[k_base].copy()
            df_u = experiments[k_ume].copy()

            # Align on snr_db
            keep_cols_b = ["snr_db"] + [m for m in metrics if m in df_b.columns]
            keep_cols_u = ["snr_db"] + [m for m in metrics if m in df_u.columns]
            df_b = df_b[keep_cols_b]
            df_u = df_u[keep_cols_u]

            merged = pd.merge(df_b, df_u, on="snr_db", how="inner", suffixes=("_base", "_ume"))
            if merged.empty:
                continue

            for m in metrics:
                col_b = f"{m}_base"
                col_u = f"{m}_ume"
                if col_b not in merged.columns or col_u not in merged.columns:
                    continue

                b = merged[col_b].astype(float).to_numpy()
                u = merged[col_u].astype(float).to_numpy()
                snr = merged["snr_db"].astype(float).to_numpy()

                # avoid invalid
                valid = np.isfinite(b) & np.isfinite(u)
                b = b[valid]; u = u[valid]; snr = snr[valid]
                if b.size == 0:
                    continue

                if m == "psnr_mean":
                    # MSE reduction(%): (mse_base - mse_ume)/mse_base
                    mse_b = _psnr_db_to_rel_mse(b)
                    mse_u = _psnr_db_to_rel_mse(u)
                    denom = np.where(mse_b == 0, np.nan, mse_b)
                    imp = (mse_b - mse_u) / denom * 100.0
                    aux_db_gain = (u - b)  # optional
                    for s, bb, uu, ii, gg in zip(snr, b, u, imp, aux_db_gain):
                        rows.append({
                            "base": base_name,
                            "ume": ume_name,
                            "channel": ch,
                            "snr_db": s,
                            "metric": m,
                            "base_value": bb,
                            "ume_value": uu,
                            "improvement_pct": ii,
                            "psnr_gain_db": gg,
                        })
                else:
                    denom = np.where(b == 0, np.nan, b)
                    if m in LOWER_IS_BETTER:
                        imp = (b - u) / denom * 100.0
                    else:
                        imp = (u - b) / denom * 100.0

                    for s, bb, uu, ii in zip(snr, b, u, imp):
                        rows.append({
                            "base": base_name,
                            "ume": ume_name,
                            "channel": ch,
                            "snr_db": s,
                            "metric": m,
                            "base_value": bb,
                            "ume_value": uu,
                            "improvement_pct": ii,
                            "psnr_gain_db": np.nan,
                        })

    return pd.DataFrame(rows)


def print_ume_improvements(result_df: pd.DataFrame):
    if result_df.empty:
        print("No matching *_UME vs base pairs found (or no overlapping snr_db points).")
        return

    # Pretty names
    metric_name_map = {
        "psnr_mean": "PSNR (MSE↓, %, + also dB gain)",
        "msssim_mean": "MS-SSIM (↑, %)",
        "lpips_mean": "LPIPS (↓, %)",
    }

    for (base, ume, ch), g in result_df.groupby(["base", "ume", "channel"], sort=True):
        print("\n" + "=" * 80)
        print(f"[{ch}] Improvement of {ume} over {base}")
        print("=" * 80)

        # pivot-like per SNR
        snrs = sorted(g["snr_db"].unique().tolist())
        metrics = sorted(g["metric"].unique().tolist())

        # Print per metric tables
        for m in metrics:
            gg = g[g["metric"] == m].sort_values("snr_db")
            title = metric_name_map.get(m, m)
            print(f"\n- {title}")
            if m == "psnr_mean":
                out = gg[["snr_db", "base_value", "ume_value", "psnr_gain_db", "improvement_pct"]].copy()
                out.columns = ["snr_db", "psnr_base_db", "psnr_ume_db", "psnr_gain_db", "mse_reduction_%"]
                print(out.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
                print(f"  Mean MSE reduction (%): {np.nanmean(out['mse_reduction_%'].to_numpy()):.4f}")
                print(f"  Mean PSNR gain (dB):    {np.nanmean(out['psnr_gain_db'].to_numpy()):.4f}")
            else:
                out = gg[["snr_db", "base_value", "ume_value", "improvement_pct"]].copy()
                out.columns = ["snr_db", "base", "ume", "improvement_%"]
                print(out.to_string(index=False, float_format=lambda x: f"{x:8.6f}"))
                print(f"  Mean improvement (%): {np.nanmean(out['improvement_%'].to_numpy()):.4f}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="metrics_*.csv 파일들을 찾을 상위 폴더")
    parser.add_argument("--out_dir", type=str, default="plots", help="그래프 이미지를 저장할 폴더")
    parser.add_argument("--use_std", action="store_true", help="켜면 *_std 컬럼으로 에러바를 함께 그림")

    # NEW: diff-only mode
    parser.add_argument(
        "--diff_only",
        action="store_true",
        help="켜면 그래프를 그리지 않고 *_UME 대비 naive(base) 개선(%)만 출력",
    )
    parser.add_argument(
        "--diff_save_csv",
        type=str,
        default="",
        help="diff_only 결과를 CSV로 저장할 경로 (예: improvements.csv). 비우면 저장 안 함.",
    )
    args = parser.parse_args()

    csv_paths = find_metric_csvs(args.root_dir)
    if not csv_paths:
        print(f"No CSV files found under {args.root_dir}")
        return

    print("Found CSV files:")
    for p in csv_paths:
        print("  ", p)

    experiments = load_experiments(csv_paths)

    metric_list = [
        ("psnr_mean", "PSNR (dB)"),
        ("msssim_mean", "MS-SSIM"),
        ("lpips_mean", "LPIPS"),
    ]
    metric_keys = [m for (m, _) in metric_list]

    # -------------------------
    # NEW: diff_only branch
    # -------------------------
    if args.diff_only:
        pairs = find_ume_pairs(experiments)
        if not pairs:
            print("No *_UME experiments found that have a matching base experiment.")
            return

        res = compute_ume_improvements(experiments, pairs, metrics=metric_keys)
        print_ume_improvements(res)

        if args.diff_save_csv:
            os.makedirs(os.path.dirname(args.diff_save_csv) or ".", exist_ok=True)
            res.to_csv(args.diff_save_csv, index=False)
            print(f"\nSaved diff CSV to: {args.diff_save_csv}")
        return

    # -------------------------
    # Original plotting branch
    # -------------------------
    for metric, metric_pretty in metric_list:
        print(f"Plotting {metric} ...")
        plot_metric_per_channel(
            experiments,
            metric,
            metric_pretty,
            args.out_dir,
            use_std=args.use_std,
        )

    print(f"Done. Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()