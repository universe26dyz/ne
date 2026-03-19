#!/usr/bin/env python3
"""Compare train-time forward intensity vs export-time density query under the same checkpoint.

Usage example:
python tools/debug/compare_train_export_forward.py \
  --checkpoint /data/dengyz/dataset/DYL/K_ne_output_v1/debug/model_iter_80000.pt \
  --config /data/dengyz/dataset/DYL/K_ne_output_v1/debug/config.json \
  --dicom-dir /data/dengyz/dataset/DYL/cine_dicom \
  --slice-index 0 \
  --n-points 2048 \
  --export-spatial-res 128,128,64

"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nesvor.pre_dcm_DYL import load_cine_dicom_dataset
from nesvor.transform import RigidTransform, ax_transform_points


def _unit_orthogonal_basis_from_affine(r_world: torch.Tensor) -> torch.Tensor:
    """Match DynamicMRIDataset slice_R construction: direction-only orthonormal basis."""
    c0 = r_world[:, 0]
    c1 = r_world[:, 1]
    e0 = c0 / (torch.norm(c0) + 1e-8)
    e1_raw = c1 / (torch.norm(c1) + 1e-8)
    e2 = torch.cross(e0, e1_raw, dim=0)
    e2 = e2 / (torch.norm(e2) + 1e-8)
    e1 = torch.cross(e2, e0, dim=0)
    e1 = e1 / (torch.norm(e1) + 1e-8)
    return torch.stack([e0, e1, e2], dim=-1)


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


def _metric_report(a: torch.Tensor, b: torch.Tensor, name: str) -> Dict[str, float]:
    d = (a - b).detach().float().cpu()
    mae = d.abs().mean().item()
    rmse = torch.sqrt((d * d).mean()).item()
    max_abs = d.abs().max().item()
    p95 = torch.quantile(d.abs(), 0.95).item()
    p99 = torch.quantile(d.abs(), 0.99).item()
    rel_mae = mae / (a.detach().abs().mean().item() + 1e-8)

    ac = a.detach().float().cpu()
    bc = b.detach().float().cpu()
    corr = float("nan")
    if ac.numel() > 1 and torch.isfinite(ac).all() and torch.isfinite(bc).all():
        corr = torch.corrcoef(torch.stack([ac, bc]))[0, 1].item()

    print(f"[{name}] MAE={mae:.6e}, RMSE={rmse:.6e}, MaxAbs={max_abs:.6e}, P95={p95:.6e}, P99={p99:.6e}, RelMAE={rel_mae:.6f}, Corr={corr:.6f}")
    return {
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "p95": p95,
        "p99": p99,
        "rel_mae": rel_mae,
        "corr": corr,
    }


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

    img = np.asarray(images[slice_index], dtype=np.float32)
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

    local = np.stack([i_sel, j_sel, np.zeros_like(i_sel)], axis=-1)  # [N,3], current code convention
    r = aff[:3, :3]
    t = aff[:3, 3]
    xyz_world = local @ r.T + t

    xyz = torch.from_numpy(xyz_world).to(device=device, dtype=torch.float32)
    xyz = (xyz - center.to(device=device, dtype=torch.float32)) / spatial_scaling
    tt = torch.full((xyz.shape[0], 1), t_val, dtype=torch.float32, device=device)
    xyzt = torch.cat([xyz, tt], dim=-1)

    slice_idx = torch.full((xyzt.shape[0],), slice_index, dtype=torch.long, device=device)
    r_world = torch.from_numpy(r).to(device=device, dtype=torch.float32)
    slice_r = _unit_orthogonal_basis_from_affine(r_world).unsqueeze(0).expand(xyzt.shape[0], 3, 3)
    return xyzt, slice_idx, slice_r


@torch.no_grad()
def query_train_like(
    model,
    xyzt: torch.Tensor,
    slice_idx: torch.Tensor,
    slice_r: torch.Tensor,
    standard_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    n = xyzt.shape[0]
    s = int(model.args.n_samples)

    xyz = xyzt[:, :3]
    t = xyzt[:, 3:]

    if standard_noise is None:
        xyz_psf_local = torch.randn(n, s, 3, device=xyzt.device, dtype=xyzt.dtype)
    else:
        if standard_noise.shape != (n, s, 3):
            raise ValueError(
                f"standard_noise shape mismatch, expected {(n, s, 3)}, got {tuple(standard_noise.shape)}"
            )
        xyz_psf_local = standard_noise
    psf_sigma = model.psf_sigma.to(xyzt.device)[slice_idx]
    xyz_psf_local = xyz_psf_local * psf_sigma.unsqueeze(1)
    xyz_psf_world = torch.einsum("bij,bsj->bsi", slice_r, xyz_psf_local)

    t_pose = model.axisangle[slice_idx][:, None]
    xyz_transformed = ax_transform_points(t_pose, xyz[:, None] + xyz_psf_world, model.trans_first)

    t_expand = t.unsqueeze(1).expand(n, s, 1)
    xyzt_sampled = torch.cat([xyz_transformed, t_expand], dim=-1)

    if model.args.n_features_slice:
        se = model.slice_embedding(slice_idx)[:, None].expand(-1, s, -1)
    else:
        se = None

    out = model.net_forward(xyzt_sampled, se)
    density = out["density"].view(n, s)
    bias = out["log_bias"].view(n, s).exp() if "log_bias" in out else torch.ones_like(density)

    if not model.args.no_slice_scale:
        c = F.softmax(model.logit_coef, 0)[slice_idx] * model.n_slices
    else:
        c = torch.ones((n,), dtype=density.dtype, device=density.device)

    v = (bias * density).mean(-1) * c
    return v


@torch.no_grad()
def query_trainlike_psf_density_only(
    model,
    xyzt: torch.Tensor,
    slice_idx: torch.Tensor,
    slice_r: torch.Tensor,
    standard_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PSF average + density only (no pose, no bias, no slice-scale, se=0)."""
    n = xyzt.shape[0]
    s = int(model.args.n_samples)
    xyz = xyzt[:, :3]
    t = xyzt[:, 3:]

    if standard_noise is None:
        xyz_psf_local = torch.randn(n, s, 3, device=xyzt.device, dtype=xyzt.dtype)
    else:
        if standard_noise.shape != (n, s, 3):
            raise ValueError(
                f"standard_noise shape mismatch, expected {(n, s, 3)}, got {tuple(standard_noise.shape)}"
            )
        xyz_psf_local = standard_noise
    psf_sigma = model.psf_sigma.to(xyzt.device)[slice_idx]
    xyz_psf_local = xyz_psf_local * psf_sigma.unsqueeze(1)
    xyz_psf_world = torch.einsum("bij,bsj->bsi", slice_r, xyz_psf_local)

    t_expand = t.unsqueeze(1).expand(n, s, 1)
    xyzt_sampled = torch.cat([xyz[:, None] + xyz_psf_world, t_expand], dim=-1)

    if model.args.n_features_slice > 0:
        se = torch.zeros(
            (n, s, model.args.n_features_slice),
            device=xyzt.device,
            dtype=xyzt.dtype,
        )
    else:
        se = None

    out = model.net_forward(xyzt_sampled, se)
    density = out["density"].view(n, s)
    return density.mean(-1)


@torch.no_grad()
def query_old_export_like(model, xyzt: torch.Tensor) -> torch.Tensor:
    if model.args.n_features_slice > 0:
        se = torch.zeros((xyzt.shape[0], model.args.n_features_slice), device=xyzt.device, dtype=xyzt.dtype)
    else:
        se = None
    out = model.net_forward(xyzt, se=se)
    return out["density"].reshape(-1)


def parse_spatial_res(s: str) -> Tuple[int, int, int]:
    vals = [v.strip() for v in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"--export-spatial-res expects 3 ints, got: {s}")
    x, y, z = [int(v) for v in vals]
    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError(f"--export-spatial-res must be positive, got: {s}")
    return x, y, z


def _resolve_output_psf_params(model, output_psf_samples: int | None, output_psf_sigma_scale: float | None):
    if output_psf_samples is None:
        samples = int(getattr(model.args, "output_psf_samples", model.args.n_samples))
    else:
        samples = int(output_psf_samples)
    samples = max(1, samples)

    if output_psf_sigma_scale is None:
        sigma_scale = float(getattr(model.args, "output_psf_sigma_scale", 0.5))
    else:
        sigma_scale = float(output_psf_sigma_scale)
    return samples, sigma_scale


def _compute_voxel_output_sigma(
    model,
    device: torch.device,
    dtype: torch.dtype,
    export_spatial_res: Tuple[int, int, int],
    output_psf_sigma_scale: float,
) -> torch.Tensor:
    bb_4d = model.inr.bounding_box.to(device=device, dtype=dtype)
    min_xyz = bb_4d[0, :3]
    max_xyz = bb_4d[1, :3]
    denom = torch.tensor(
        [
            max(export_spatial_res[0] - 1, 1),
            max(export_spatial_res[1] - 1, 1),
            max(export_spatial_res[2] - 1, 1),
        ],
        dtype=dtype,
        device=device,
    )
    voxel_size = (max_xyz - min_xyz) / denom
    return voxel_size * output_psf_sigma_scale


def _compute_train_psf_output_sigma(
    model,
    slice_idx: torch.Tensor,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    psf_sigma = model.psf_sigma.to(device=device, dtype=dtype)
    if mode == "slice":
        sigma = psf_sigma[slice_idx]
        return sigma.mean(dim=0)
    if mode == "mean":
        return psf_sigma.mean(dim=0)
    raise ValueError(f"Unsupported train-psf kernel mode: {mode}")


@torch.no_grad()
def query_new_export_like(
    model,
    xyzt: torch.Tensor,
    output_sigma: torch.Tensor,
    output_psf_samples: int,
    standard_noise: Optional[torch.Tensor] = None,
    slice_r: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mirror train.py::render_and_save_nifti query logic on arbitrary sampled points."""
    chunk_size = int(getattr(model.args, "chunk_size", 200000))
    effective_chunk_size = max(1, chunk_size // output_psf_samples)

    densities = []
    for i in range(0, xyzt.shape[0], effective_chunk_size):
        chunk = xyzt[i : i + effective_chunk_size]
        if output_psf_samples > 1:
            xyz_chunk = chunk[:, :3]
            t_chunk = chunk[:, 3:]
            if standard_noise is None:
                base_noise = torch.randn(
                    chunk.shape[0], output_psf_samples, 3,
                    dtype=chunk.dtype, device=chunk.device
                )
            else:
                base_noise = standard_noise[i : i + chunk.shape[0]]
                if base_noise.shape != (chunk.shape[0], output_psf_samples, 3):
                    raise ValueError(
                        "standard_noise shape mismatch in query_new_export_like"
                    )
            noise_local = base_noise * output_sigma[None, None, :]
            if slice_r is not None:
                rot_chunk = slice_r[i : i + chunk.shape[0]]
                noise_world = torch.einsum("bij,bsj->bsi", rot_chunk, noise_local)
            else:
                noise_world = noise_local
            xyz_jitter = xyz_chunk[:, None, :] + noise_world
            t_jitter = t_chunk[:, None, :].expand(-1, output_psf_samples, -1)
            query = torch.cat([xyz_jitter, t_jitter], dim=-1).reshape(-1, 4)
        else:
            query = chunk

        if model.args.n_features_slice > 0:
            se_neutral = torch.zeros(
                (query.shape[0], model.args.n_features_slice),
                dtype=query.dtype,
                device=query.device,
            )
        else:
            se_neutral = None

        results = model.net_forward(query, se=se_neutral)
        density_all = results["density"].reshape(-1)
        if output_psf_samples > 1:
            density = density_all.view(chunk.shape[0], output_psf_samples).mean(-1)
        else:
            density = density_all
        densities.append(density)

    return torch.cat(densities, dim=0)


@torch.no_grad()
def query_center_full(model, xyzt: torch.Tensor, slice_idx: torch.Tensor) -> torch.Tensor:
    if model.args.n_features_slice > 0:
        se = model.slice_embedding(slice_idx)
    else:
        se = None
    out = model.net_forward(xyzt, se=se)
    density = out["density"].reshape(-1)
    bias = out["log_bias"].exp().reshape(-1) if "log_bias" in out else torch.ones_like(density)

    if not model.args.no_slice_scale:
        c = F.softmax(model.logit_coef, 0)[slice_idx] * model.n_slices
    else:
        c = torch.ones_like(density)
    return density * bias * c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare train-time forward vs export-time query")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_iter_xxxxx.pt")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json used for training")
    parser.add_argument("--dicom-dir", type=str, default=None, help="Override dicom dir. Default uses config.dicom_dir")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu")
    parser.add_argument("--slice-index", type=int, default=0, help="Slice index for comparison")
    parser.add_argument("--n-points", type=int, default=2048, help="Number of random points from one slice")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument(
        "--export-spatial-res",
        type=str,
        default="128,128,64",
        help="Spatial resolution used by train.py render_and_save_nifti (x,y,z), e.g. 128,128,64",
    )
    parser.add_argument(
        "--output-psf-samples",
        type=int,
        default=None,
        help="Override output_psf_samples for NewExportLike. Default uses model.args.output_psf_samples / n_samples",
    )
    parser.add_argument(
        "--output-psf-sigma-scale",
        type=float,
        default=None,
        help="Override output_psf_sigma_scale for NewExportLike. Default uses model.args.output_psf_sigma_scale or 0.5",
    )
    parser.add_argument(
        "--train-psf-kernel-mode",
        type=str,
        default="slice",
        choices=["slice", "mean"],
        help="Kernel source for NewExportLike(train_psf_kernel): slice-specific or global mean psf_sigma",
    )
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

    device = torch.device(args.device)
    model, images, affines, timestamps, masks = build_model_and_data(args, ckpt, dicom_dir)
    export_spatial_res = parse_spatial_res(cli.export_spatial_res)
    output_psf_samples, output_psf_sigma_scale = _resolve_output_psf_params(
        model, cli.output_psf_samples, cli.output_psf_sigma_scale
    )

    xyzt, slice_idx, slice_r = sample_points_from_slice(
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

    output_sigma_voxel = _compute_voxel_output_sigma(
        model=model,
        device=xyzt.device,
        dtype=xyzt.dtype,
        export_spatial_res=export_spatial_res,
        output_psf_sigma_scale=output_psf_sigma_scale,
    )
    output_sigma_train = _compute_train_psf_output_sigma(
        model=model,
        slice_idx=slice_idx,
        mode=cli.train_psf_kernel_mode,
        device=xyzt.device,
        dtype=xyzt.dtype,
    )
    model_psf_mean = model.psf_sigma.detach().float().mean(dim=0).cpu()
    model_psf_slice = model.psf_sigma.to(device=slice_idx.device).detach().float()[slice_idx].mean(dim=0).cpu()

    with torch.no_grad():
        train_samples = int(model.args.n_samples)
        shared_noise_train = torch.randn(
            xyzt.shape[0], train_samples, 3, device=xyzt.device, dtype=xyzt.dtype
        )
        use_shared_noise_for_export = output_psf_samples == train_samples
        shared_noise_export = shared_noise_train if use_shared_noise_for_export else None

        v_train = query_train_like(
            model, xyzt, slice_idx, slice_r, standard_noise=shared_noise_train
        )
        v_train_density_psf = query_trainlike_psf_density_only(
            model, xyzt, slice_idx, slice_r, standard_noise=shared_noise_train
        )
        v_export_old = query_old_export_like(model, xyzt)
        v_export_new = query_new_export_like(
            model,
            xyzt,
            output_sigma=output_sigma_voxel,
            output_psf_samples=output_psf_samples,
            standard_noise=shared_noise_export,
        )
        v_export_new_train_psf = query_new_export_like(
            model,
            xyzt,
            output_sigma=output_sigma_train,
            output_psf_samples=output_psf_samples,
            standard_noise=shared_noise_export,
        )
        v_export_new_train_psf_rot = query_new_export_like(
            model,
            xyzt,
            output_sigma=output_sigma_train,
            output_psf_samples=output_psf_samples,
            standard_noise=shared_noise_export,
            slice_r=slice_r,
        )
        v_center_full = query_center_full(model, xyzt, slice_idx)

    print("=" * 90)
    print("[Forward Formula Summary]")
    print("Train-like : v = c * mean_s( exp(log_bias_s) * density_s ), with PSF sampling + pose transform")
    print("TrainLike_NoPose_NoBiasScale: v = mean_s( density(x + eps_s, t) ), se=0")
    print("OldExport-like: v = density(center point), se=0, no PSF, no bias, no slice-scale")
    print(
        "NewExport-like: v = mean_k( density(x + eps_k, t) ), "
        f"se=0, output_psf_samples={output_psf_samples}, output_psf_sigma_scale={output_psf_sigma_scale:.4f}, "
        f"export_spatial_res={export_spatial_res}"
    )
    print(
        "NewExport-like(train_psf_kernel): "
        f"v = mean_k( density(x + eps_k, t) ), se=0, kernel_from=train_psf_sigma({cli.train_psf_kernel_mode})"
    )
    print(
        "NewExport-like(train_psf_kernel+sliceR): "
        "v = mean_k( density(x + R_slice*eps_k, t) ), se=0"
    )
    print("Center-full: v = c * exp(log_bias_center) * density_center (no PSF)")
    print(
        f"Sigma debug: model.psf_sigma.mean={model_psf_mean.numpy()}, "
        f"model.psf_sigma.slice_mean={model_psf_slice.numpy()}"
    )
    print(
        f"Sigma debug: output_sigma(voxel_kernel)={output_sigma_voxel.detach().float().cpu().numpy()}, "
        f"output_sigma(train_psf_kernel)={output_sigma_train.detach().float().cpu().numpy()}"
    )
    print(
        f"Noise debug: shared_noise_train_shape={tuple(shared_noise_train.shape)}, "
        f"shared_noise_for_export={'ON' if use_shared_noise_for_export else 'OFF'}"
    )
    if not use_shared_noise_for_export:
        print(
            f"Noise debug: output_psf_samples({output_psf_samples}) != n_samples({train_samples}), "
            "export path uses independent noise."
        )
    print("=" * 90)

    m_train_export_old = _metric_report(v_train, v_export_old, "TrainLike vs OldExportLike")
    m_train_export_new = _metric_report(v_train, v_export_new, "TrainLike vs NewExportLike")
    m_train_export_new_trainpsf = _metric_report(
        v_train, v_export_new_train_psf, "TrainLike vs NewExportLike(train_psf_kernel)"
    )
    m_train_export_new_trainpsf_rot = _metric_report(
        v_train, v_export_new_train_psf_rot, "TrainLike vs NewExportLike(train_psf_kernel+sliceR)"
    )
    m_train_center = _metric_report(v_train, v_center_full, "TrainLike vs CenterFull(no PSF)")
    m_train_nb_new = _metric_report(
        v_train_density_psf, v_export_new, "TrainLike_NoPose_NoBiasScale vs NewExportLike"
    )
    m_train_nb_new_trainpsf = _metric_report(
        v_train_density_psf,
        v_export_new_train_psf,
        "TrainLike_NoPose_NoBiasScale vs NewExportLike(train_psf_kernel)",
    )
    m_train_nb_new_trainpsf_rot = _metric_report(
        v_train_density_psf,
        v_export_new_train_psf_rot,
        "TrainLike_NoPose_NoBiasScale vs NewExportLike(train_psf_kernel+sliceR)",
    )
    m_train_nb_old = _metric_report(
        v_train_density_psf, v_export_old, "TrainLike_NoPose_NoBiasScale vs OldExportLike"
    )
    m_old_vs_new = _metric_report(v_export_old, v_export_new, "OldExportLike vs NewExportLike")
    m_new_vs_newtrainpsf = _metric_report(
        v_export_new, v_export_new_train_psf, "NewExportLike(voxel_kernel) vs NewExportLike(train_psf_kernel)"
    )
    m_newtrainpsf_vs_rot = _metric_report(
        v_export_new_train_psf,
        v_export_new_train_psf_rot,
        "NewExportLike(train_psf_kernel) vs NewExportLike(train_psf_kernel+sliceR)",
    )

    print("-" * 90)
    print("[Normal / Abnormal Criteria]")
    print("Normal (recommended): TrainLike vs NewExportLike RelMAE < 0.05 AND Corr > 0.98")
    print("Abnormal: above threshold indicates train/export inconsistency is likely material")

    abnormal_old = not (m_train_export_old["rel_mae"] < 0.05 and m_train_export_old["corr"] > 0.98)
    abnormal_new = not (m_train_export_new["rel_mae"] < 0.05 and m_train_export_new["corr"] > 0.98)
    abnormal_new_trainpsf = not (
        m_train_export_new_trainpsf["rel_mae"] < 0.05 and m_train_export_new_trainpsf["corr"] > 0.98
    )
    abnormal_new_trainpsf_rot = not (
        m_train_export_new_trainpsf_rot["rel_mae"] < 0.05 and m_train_export_new_trainpsf_rot["corr"] > 0.98
    )
    psf_gap_large = m_train_center["rel_mae"] > 0.02
    old_new_gap_large = m_old_vs_new["rel_mae"] > 0.02
    nb_new_better = m_train_nb_new["rel_mae"] < m_train_nb_old["rel_mae"]
    nb_new_trainpsf_better = m_train_nb_new_trainpsf["rel_mae"] < m_train_nb_old["rel_mae"]
    nb_new_trainpsf_rot_better = m_train_nb_new_trainpsf_rot["rel_mae"] < m_train_nb_old["rel_mae"]
    new_trainpsf_better = m_train_export_new_trainpsf["rel_mae"] < m_train_export_new["rel_mae"]
    new_trainpsf_rot_better = m_train_export_new_trainpsf_rot["rel_mae"] < m_train_export_new_trainpsf["rel_mae"]
    kernel_gap_large = m_new_vs_newtrainpsf["rel_mae"] > 0.02
    rotation_gap_large = m_newtrainpsf_vs_rot["rel_mae"] > 0.02

    print("-" * 90)
    print("[Conclusion Hint]")
    print(f"Old export inconsistency abnormal? {'YES' if abnormal_old else 'NO'}")
    print(f"New export inconsistency abnormal? {'YES' if abnormal_new else 'NO'}")
    print(f"New export(train_psf_kernel) inconsistency abnormal? {'YES' if abnormal_new_trainpsf else 'NO'}")
    print(f"New export(train_psf_kernel+sliceR) inconsistency abnormal? {'YES' if abnormal_new_trainpsf_rot else 'NO'}")
    print(f"TrainLike vs CenterFull(no PSF) gap significant? {'YES' if psf_gap_large else 'NO'}")
    print(f"Old vs New export gap significant? {'YES' if old_new_gap_large else 'NO'}")
    print(f"Voxel-kernel vs TrainPSF-kernel gap significant? {'YES' if kernel_gap_large else 'NO'}")
    print(f"TrainPSF-kernel vs TrainPSF+sliceR gap significant? {'YES' if rotation_gap_large else 'NO'}")
    print(f"New export better than old under no-pose/no-bias baseline? {'YES' if nb_new_better else 'NO'}")
    print(
        f"New export(train_psf_kernel) better than old under no-pose/no-bias baseline? "
        f"{'YES' if nb_new_trainpsf_better else 'NO'}"
    )
    print(
        f"New export(train_psf_kernel+sliceR) better than old under no-pose/no-bias baseline? "
        f"{'YES' if nb_new_trainpsf_rot_better else 'NO'}"
    )
    print(f"TrainLike vs New export: train_psf_kernel better than voxel_kernel? {'YES' if new_trainpsf_better else 'NO'}")
    print(
        f"TrainLike vs New export: train_psf_kernel+sliceR better than train_psf_kernel? "
        f"{'YES' if new_trainpsf_rot_better else 'NO'}"
    )

    if abnormal_new:
        print("提示: 即使启用 NewExportLike，导出与训练目标仍不一致，需继续排查。")
    elif abnormal_old and not abnormal_new:
        print("提示: P2 方向生效，NewExportLike 显著缩小了 train/export 不一致。")
    else:
        print("提示: Old/New 导出都与训练目标较一致（本批样本）。")
    print("=" * 90)


if __name__ == "__main__":
    main()
