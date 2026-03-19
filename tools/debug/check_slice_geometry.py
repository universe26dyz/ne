#!/usr/bin/env python3
"""Check slice geometry consistency: slice_R orthogonality, row/col mapping, spacing usage.

Usage example:
python tools/debug/check_slice_geometry.py \
  --dicom-dir /home/universe/SVR/data/DYL/cine_dicom \
  --stack cine_sax --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nesvor.pre_dcm_DYL import compute_affine_from_dicom
from nesvor.inr.data import DynamicMRIDataset


def _safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _step_vectors(affine: np.ndarray, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
    r = affine[:3, :3]
    t = affine[:3, 3]

    def world(ii: int, jj: int) -> np.ndarray:
        local = np.array([float(ii), float(jj), 0.0], dtype=np.float64)
        return local @ r.T + t

    d_i = world(i + 1, j) - world(i, j)
    d_j = world(i, j + 1) - world(i, j)
    return d_i, d_j


def _matrix_stats(
    m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    c0 = m[:, 0]
    c1 = m[:, 1]
    c2 = m[:, 2]
    norms = np.array(
        [np.linalg.norm(c0), np.linalg.norm(c1), np.linalg.norm(c2)], dtype=np.float64
    )
    d01 = float(np.dot(c0, c1))
    d02 = float(np.dot(c0, c2))
    d12 = float(np.dot(c1, c2))
    return c0, c1, c2, norms, d01, d02, d12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check slice_R / row-col / spacing consistency")
    parser.add_argument("--dicom-dir", type=str, required=True, help="Root dir containing cine_sax/cine_4ch/... stacks")
    parser.add_argument("--stack", type=str, default="cine_sax", help="Stack name (e.g., cine_sax)")
    parser.add_argument("--index", type=int, default=0, help="DICOM index in sorted order")
    parser.add_argument("--i", type=int, default=100, help="Pixel row index for local finite difference")
    parser.add_argument("--j", type=int, default=100, help="Pixel col index for local finite difference")
    parser.add_argument("--tol-norm", type=float, default=1e-3, help="Tolerance for unit-norm check")
    parser.add_argument("--tol-dot", type=float, default=1e-3, help="Tolerance for orthogonality check")
    parser.add_argument("--tol-spacing-mm", type=float, default=1e-3, help="Tolerance for spacing check in mm")
    parser.add_argument("--mask-threshold-ratio", type=float, default=0.05, help="Mask threshold ratio for single-slice DynamicMRIDataset instantiation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stack_dir = Path(args.dicom_dir) / args.stack
    files = sorted(stack_dir.rglob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM found under: {stack_dir}")
    if args.index < 0 or args.index >= len(files):
        raise IndexError(f"index={args.index} out of range [0, {len(files)-1}]")

    dcm_path = files[args.index]
    ds = pydicom.dcmread(str(dcm_path))

    affine = compute_affine_from_dicom(ds)
    r = affine[:3, :3]

    row_cos = np.array([float(x) for x in ds.ImageOrientationPatient[:3]], dtype=np.float64)
    col_cos = np.array([float(x) for x in ds.ImageOrientationPatient[3:6]], dtype=np.float64)
    normal_cos = np.cross(row_cos, col_cos)

    if "PixelSpacing" not in ds:
        raise KeyError("PixelSpacing not found in DICOM header")
    row_spacing = float(ds.PixelSpacing[0])
    col_spacing = float(ds.PixelSpacing[1])

    i = min(max(args.i, 0), int(ds.Rows) - 2)
    j = min(max(args.j, 0), int(ds.Columns) - 2)
    d_i, d_j = _step_vectors(affine, i=i, j=j)

    c0, c1, c2, norms_affine, d01, d02, d12 = _matrix_stats(r)
    n0, n1, n2 = norms_affine.tolist()

    # 额外：实例化 DynamicMRIDataset，并读取真正供 PSF 使用的 dataset.slice_R
    img = ds.pixel_array.astype(np.float32)
    mask = img > (img.max() * args.mask_threshold_ratio)
    if not mask.any():
        mask = np.ones_like(img, dtype=bool)
    ds_runtime = DynamicMRIDataset(
        images_list=[img],
        affines_list=[affine.astype(np.float32)],
        timestamps_list=[0.0],
        masks_list=[mask],
    )
    psf_r = ds_runtime.slice_R[0].detach().cpu().numpy().astype(np.float64)
    p0, p1, p2, norms_psf, p01, p02, p12 = _matrix_stats(psf_r)

    expected_i_correct = col_cos * row_spacing  # i=row index increments
    expected_j_correct = row_cos * col_spacing  # j=col index increments

    expected_i_impl = col_cos * col_spacing  # current implementation tendency
    expected_j_impl = row_cos * row_spacing

    print("=" * 80)
    print("[Slice Geometry Check]")
    print(f"DICOM: {dcm_path}")
    print(f"Rows x Cols: {int(ds.Rows)} x {int(ds.Columns)}")
    print(f"PixelSpacing (row, col) = ({row_spacing:.6f}, {col_spacing:.6f})")
    print("-" * 80)
    print("ImageOrientationPatient:")
    print(f"  row_cos   = {row_cos}")
    print(f"  col_cos   = {col_cos}")
    print(f"  normal    = {normal_cos}")
    print("-" * 80)
    print("[World Mapping Matrix] affine[:3,:3] columns:")
    print(f"  c0 = {c0}, ||c0||={n0:.6f}")
    print(f"  c1 = {c1}, ||c1||={n1:.6f}")
    print(f"  c2 = {c2}, ||c2||={n2:.6f}")
    print(f"  dot(c0,c1)={d01:.6e}, dot(c0,c2)={d02:.6e}, dot(c1,c2)={d12:.6e}")
    print("-" * 80)
    print("[PSF Rotation Matrix] dataset.slice_R columns (from DynamicMRIDataset):")
    print(f"  p0 = {p0}, ||p0||={norms_psf[0]:.6f}")
    print(f"  p1 = {p1}, ||p1||={norms_psf[1]:.6f}")
    print(f"  p2 = {p2}, ||p2||={norms_psf[2]:.6f}")
    print(f"  dot(p0,p1)={p01:.6e}, dot(p0,p2)={p02:.6e}, dot(p1,p2)={p12:.6e}")
    print("-" * 80)
    print(f"Finite-diff world steps at (i={i}, j={j}) under current world= [i,j,0] @ R^T + T:")
    print(f"  d_i = world(i+1,j)-world(i,j) = {d_i}, |d_i|={np.linalg.norm(d_i):.6f}")
    print(f"  d_j = world(i,j+1)-world(i,j) = {d_j}, |d_j|={np.linalg.norm(d_j):.6f}")
    print("  cos(d_i,row_cos)=%.6f  cos(d_i,col_cos)=%.6f" % (_safe_cos(d_i, row_cos), _safe_cos(d_i, col_cos)))
    print("  cos(d_j,row_cos)=%.6f  cos(d_j,col_cos)=%.6f" % (_safe_cos(d_j, row_cos), _safe_cos(d_j, col_cos)))
    print("-" * 80)
    print("Expected if i=row, j=col (recommended for your dataset indexing):")
    print(f"  expected_i_correct = col_cos * row_spacing = {expected_i_correct}")
    print(f"  expected_j_correct = row_cos * col_spacing = {expected_j_correct}")
    print("Current implementation-like expectation:")
    print(f"  expected_i_impl = col_cos * col_spacing = {expected_i_impl}")
    print(f"  expected_j_impl = row_cos * row_spacing = {expected_j_impl}")
    print("-" * 80)

    is_unit_rotation = (
        abs(n0 - 1.0) < args.tol_norm
        and abs(n1 - 1.0) < args.tol_norm
        and abs(n2 - 1.0) < args.tol_norm
        and abs(d01) < args.tol_dot
        and abs(d02) < args.tol_dot
        and abs(d12) < args.tol_dot
    )
    is_unit_psf_rotation = (
        abs(norms_psf[0] - 1.0) < args.tol_norm
        and abs(norms_psf[1] - 1.0) < args.tol_norm
        and abs(norms_psf[2] - 1.0) < args.tol_norm
        and abs(p01) < args.tol_dot
        and abs(p02) < args.tol_dot
        and abs(p12) < args.tol_dot
    )

    spacing_ok_recommended = (
        abs(np.linalg.norm(d_i) - row_spacing) < args.tol_spacing_mm
        and abs(np.linalg.norm(d_j) - col_spacing) < args.tol_spacing_mm
    )

    spacing_matches_impl = (
        abs(np.linalg.norm(d_i) - col_spacing) < args.tol_spacing_mm
        and abs(np.linalg.norm(d_j) - row_spacing) < args.tol_spacing_mm
    )

    map_i_to_col = _safe_cos(d_i, col_cos) > 0.99
    map_j_to_row = _safe_cos(d_j, row_cos) > 0.99

    print("[Judgement]")
    print(f"  Rotation quality (normal expects unit+orthogonal): {'NORMAL' if is_unit_rotation else 'ABNORMAL'}")
    print(f"  PSF rotation quality (dataset.slice_R expects unit+orthogonal): {'NORMAL' if is_unit_psf_rotation else 'ABNORMAL'}")
    print(f"  Mapping i->col and j->row: {'YES' if (map_i_to_col and map_j_to_row) else 'NO'}")
    print(f"  Spacing consistent with recommended (i=row->row_spacing, j=col->col_spacing): {'YES' if spacing_ok_recommended else 'NO'}")
    print(f"  Spacing consistent with current impl-like swapped usage: {'YES' if spacing_matches_impl else 'NO'}")
    if abs(row_spacing - col_spacing) < args.tol_spacing_mm:
        print("  Note: row_spacing ~= col_spacing, this slice alone cannot disambiguate spacing swap.")

    print("-" * 80)
    if not is_unit_rotation:
        print("结论提示: slice_R 含有尺度成分，不是单位旋转矩阵。若直接用于 PSF 方向旋转，属于异常风险。")
    if (not is_unit_rotation) and is_unit_psf_rotation:
        print("P1 判定: 生效（world mapping matrix 含尺度，但 dataset.slice_R 已为单位正交）。")
    elif not is_unit_psf_rotation:
        print("P1 判定: 未生效（dataset.slice_R 仍带尺度或非正交）。")
    else:
        print("P1 判定: 部分生效或无需该修复（请结合具体数据判断）。")
    if map_i_to_col and map_j_to_row and spacing_matches_impl and not spacing_ok_recommended:
        print("结论提示: 方向映射看似正确，但 spacing 在 i/j 上存在交换风险（异常）。")
    elif spacing_ok_recommended and is_unit_rotation:
        print("结论提示: 几何关系基本正常。")
    else:
        print("结论提示: 几何链路存在可疑项，请结合输出逐项修复。")
    print("=" * 80)


if __name__ == "__main__":
    main()
