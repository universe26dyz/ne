#!/usr/bin/env python3
"""Check whether bias regularization mean(log_bias)^2 is ineffective.

Usage example:
python tools/debug/check_bias_regularization.py \
  --checkpoint /data/dengyz/dataset/DYL/K_ne_output_v1/debug/model_iter_80000.pt \
  --config /data/dengyz/dataset/DYL/K_ne_output_v1/debug/config.json \
  --dicom-dir /data/dengyz/dataset/DYL/cine_dicom \
  --slice-index 0 --n-points 4096
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nesvor.pre_dcm_DYL import load_cine_dicom_dataset
from nesvor.transform import RigidTransform


def load_nesvor_class():
    try:
        from nesvor.inr.models import NeSVoR  # noqa: WPS433
    except ModuleNotFoundError as e:
        if "tinycudann" in str(e):
            print("[Dependency Error] tinycudann is missing.")
            print("This script needs tinycudann-compatible model loading for your checkpoint.")
            print("Please run this script in the same environment used for training, or install tinycudann first.")
            raise SystemExit(2) from e
        raise
    return NeSVoR


def parse_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).strip()
    if s in {"torch.float16", "float16", "fp16", "half"}:
        return torch.float16
    if s in {"torch.float32", "float32", "fp32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


def namespace_from_config(config_path: Path, device_override: str | None) -> Namespace:
    cfg = json.loads(config_path.read_text())
    if "dtype" in cfg:
        cfg["dtype"] = parse_dtype(cfg["dtype"])
    cfg.setdefault("delta", 0.1)
    cfg.setdefault("n_features_per_level", 2)
    cfg.setdefault("weight_image", 1.0)
    cfg.setdefault("img_reg_autodiff", False)
    if device_override is not None:
        cfg["device"] = device_override
    return Namespace(**cfg)


def build_model_and_data(args: Namespace, ckpt: Dict, dicom_dir: str):
    NeSVoR = load_nesvor_class()
    images, affines, timestamps, masks = load_cine_dicom_dataset(dicom_dir)
    if len(images) == 0:
        raise RuntimeError("No DICOM loaded. Check --dicom-dir")

    n_slices = len(images)
    spatial_scaling = float(ckpt.get("spatial_scaling", 30.0))
    resolution_tensor = (
        torch.tensor(args.psf_resolution, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n_slices, 3)
        / spatial_scaling
    )

    bb_dummy = torch.tensor(
        [[-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    init_trans = RigidTransform(torch.zeros(n_slices, 6, dtype=torch.float32))

    model = NeSVoR(
        transformation=init_trans,
        resolution=resolution_tensor,
        v_mean=0.5,
        bounding_box=bb_dummy,
        spatial_scaling=spatial_scaling,
        args=args,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, images, affines, timestamps, masks


def sample_points_from_slice(
    images,
    affines,
    timestamps,
    masks,
    slice_index: int,
    n_points: int,
    center: torch.Tensor,
    spatial_scaling: float,
    device: torch.device,
    seed: int,
):
    if slice_index < 0 or slice_index >= len(images):
        raise IndexError(f"slice-index out of range: {slice_index}")

    aff = np.asarray(affines[slice_index], dtype=np.float32)
    t_val = float(timestamps[slice_index])
    mask = np.asarray(masks[slice_index], dtype=bool)

    ii, jj = np.where(mask)
    if ii.size == 0:
        raise RuntimeError(f"No valid masked pixel in slice {slice_index}")

    rng = np.random.default_rng(seed)
    pick = rng.choice(ii.size, size=min(n_points, ii.size), replace=False)
    i_sel = ii[pick].astype(np.float32)
    j_sel = jj[pick].astype(np.float32)

    local = np.stack([i_sel, j_sel, np.zeros_like(i_sel)], axis=-1)
    r = aff[:3, :3]
    t = aff[:3, 3]
    xyz_world = local @ r.T + t

    xyz = torch.from_numpy(xyz_world).to(device=device, dtype=torch.float32)
    xyz = (xyz - center.to(device=device, dtype=torch.float32)) / spatial_scaling
    tt = torch.full((xyz.shape[0], 1), t_val, dtype=torch.float32, device=device)
    xyzt = torch.cat([xyz, tt], dim=-1)

    slice_idx = torch.full((xyzt.shape[0],), slice_index, dtype=torch.long, device=device)
    return xyzt, slice_idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check bias regularization effectiveness")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_iter_xxxxx.pt")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json used for training")
    parser.add_argument("--dicom-dir", type=str, default=None, help="Override dicom dir. Default uses config.dicom_dir")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu")
    parser.add_argument("--slice-index", type=int, default=0, help="Slice index for checking bias")
    parser.add_argument("--n-points", type=int, default=4096, help="Number of random points sampled from one slice")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    ckpt_path = Path(cli.checkpoint)
    cfg_path = Path(cli.config)

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    args = namespace_from_config(cfg_path, cli.device)
    if "cuda" in str(args.device) and not torch.cuda.is_available():
        print("[Device Error] Config/device requires CUDA, but CUDA is not available.")
        print("Use --device cpu only if your checkpoint/model stack supports CPU inference.")
        raise SystemExit(2)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    center = ckpt.get("center", torch.zeros(3, dtype=torch.float32))
    if not isinstance(center, torch.Tensor):
        center = torch.tensor(center, dtype=torch.float32)
    spatial_scaling = float(ckpt.get("spatial_scaling", 30.0))

    dicom_dir = cli.dicom_dir if cli.dicom_dir is not None else getattr(args, "dicom_dir", None)
    if dicom_dir is None:
        raise ValueError("dicom_dir is missing. Provide --dicom-dir or include it in config.json")

    model, images, affines, timestamps, masks = build_model_and_data(args, ckpt, dicom_dir)

    if not getattr(model.args, "n_levels_bias", 0):
        print("[Info] n_levels_bias=0, bias branch is disabled. Script exits.")
        return

    device = torch.device(args.device)
    xyzt, slice_idx = sample_points_from_slice(
        images=images,
        affines=affines,
        timestamps=timestamps,
        masks=masks,
        slice_index=cli.slice_index,
        n_points=cli.n_points,
        center=center,
        spatial_scaling=spatial_scaling,
        device=device,
        seed=cli.seed,
    )

    with torch.no_grad():
        if model.args.n_features_slice > 0:
            se = model.slice_embedding(slice_idx)
        else:
            se = None
        out = model.net_forward(xyzt, se=se)

    if "log_bias" not in out:
        print("[Info] log_bias not found in model outputs. Bias branch may be disabled.")
        return

    log_bias = out["log_bias"].reshape(-1).detach().float().cpu()

    mean_v = log_bias.mean().item()
    std_v = log_bias.std(unbiased=False).item()
    maxabs_v = log_bias.abs().max().item()
    p95 = torch.quantile(log_bias.abs(), 0.95).item()
    p99 = torch.quantile(log_bias.abs(), 0.99).item()

    old_reg = (log_bias.mean() ** 2).item()          # current code
    new_reg = (log_bias.pow(2).mean()).item()        # minimal fix target
    ratio = new_reg / (old_reg + 1e-12)

    print("=" * 90)
    print("[Bias Regularization Check]")
    print("Current reg in training code: biasReg = mean(log_bias)^2")
    print("Alternative robust reg       : mean(log_bias^2)")
    print("-" * 90)
    print(f"N={log_bias.numel()}")
    print(f"log_bias mean={mean_v:.6e}, std={std_v:.6e}, maxabs={maxabs_v:.6e}, |p95|={p95:.6e}, |p99|={p99:.6e}")
    print(f"old_reg = mean(log_bias)^2   = {old_reg:.6e}")
    print(f"new_reg = mean(log_bias^2)   = {new_reg:.6e}")
    print(f"new/old ratio                = {ratio:.6e}")

    print("-" * 90)
    print("[Normal / Abnormal Criteria]")
    print("Normal: new/old ratio close to 1~3 and std is small")
    print("Abnormal: new/old ratio >> 10 while std/maxabs not small (means old reg only suppresses mean)")

    abnormal = (ratio > 10.0 and std_v > 1e-3) or (abs(mean_v) < 1e-4 and std_v > 1e-2)
    print("-" * 90)
    print(f"Judgement: {'ABNORMAL (likely ineffective biasReg)' if abnormal else 'NORMAL-ish'}")
    if abnormal:
        print("结论提示: 当前 biasReg 很可能只压均值，不压振幅。建议最小修复为 mean(log_bias^2)。")
    else:
        print("结论提示: 当前样本下 biasReg 未表现出明显失效。")
    print("=" * 90)


if __name__ == "__main__":
    main()
