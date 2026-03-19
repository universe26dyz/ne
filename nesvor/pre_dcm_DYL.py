import os
import re
import glob
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import torch

from .image import Stack
from .image.image_utils import transformation2affine
from .svort.inference import svort_predict
from .transform import RigidTransform


def compute_affine_from_dicom(ds):
    """
    根据 DICOM 头文件计算 2D 切片到物理世界坐标的 4x4 仿射矩阵
    """
    iop = np.array([float(x) for x in ds.ImageOrientationPatient], dtype=np.float64)
    row_cosine = iop[0:3]
    col_cosine = iop[3:6]

    ipp = np.array([float(x) for x in ds.ImagePositionPatient], dtype=np.float64)

    if "PixelSpacing" in ds:
        ps = np.array([float(x) for x in ds.PixelSpacing], dtype=np.float64)
        dy, dx = float(ps[0]), float(ps[1])
    else:
        dy, dx = 1.0, 1.0

    normal = np.cross(row_cosine, col_cosine)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    affine = np.eye(4, dtype=np.float64)
    # i=row index 的步进向量：列方向余弦 * row spacing(dy)
    affine[:3, 0] = col_cosine * dy
    # j=col index 的步进向量：行方向余弦 * col spacing(dx)
    affine[:3, 1] = row_cosine * dx
    # 切片法线方向
    affine[:3, 2] = normal
    affine[:3, 3] = ipp
    return affine


def generate_background_mask(image_array, threshold_ratio=0.05):
    """
    简单阈值掩膜，剔除空气背景
    """
    max_val = float(image_array.max()) if image_array.size else 0.0
    if max_val <= 0:
        return np.zeros_like(image_array, dtype=bool)
    return image_array > (max_val * threshold_ratio)


def _default_stacks(stack_names: Optional[Sequence[str]]) -> List[str]:
    if stack_names is None:
        return ["cine_sax", "cine_4ch", "cine_2ch"]
    return list(stack_names)


def _parse_frame_idx(name: str) -> Optional[int]:
    m = re.search(r"(\d+)$", name)
    if m is None:
        return None
    return int(m.group(1))


def _read_dicom_geometry(ds) -> Dict[str, Any]:
    iop = np.array([float(x) for x in ds.ImageOrientationPatient], dtype=np.float64)
    row_cos = iop[0:3]
    col_cos = iop[3:6]
    row_cos = row_cos / (np.linalg.norm(row_cos) + 1e-8)
    col_cos = col_cos / (np.linalg.norm(col_cos) + 1e-8)
    normal = np.cross(row_cos, col_cos)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    ipp = np.array([float(x) for x in ds.ImagePositionPatient], dtype=np.float64)
    if "PixelSpacing" in ds:
        ps = [float(x) for x in ds.PixelSpacing]
        dy, dx = float(ps[0]), float(ps[1])
    else:
        dy, dx = 1.0, 1.0

    if "SliceThickness" in ds:
        thickness = float(ds.SliceThickness)
    elif "SpacingBetweenSlices" in ds:
        thickness = float(ds.SpacingBetweenSlices)
    else:
        thickness = 1.0

    return {
        "row_cos": row_cos,
        "col_cos": col_cos,
        "normal": normal,
        "ipp": ipp,
        "dx": dx,
        "dy": dy,
        "thickness": thickness,
    }


def _slice_transform_matrix(
    geom: Dict[str, Any], height: int, width: int
) -> np.ndarray:
    row_cos = geom["row_cos"]
    col_cos = geom["col_cos"]
    normal = geom["normal"]
    ipp = geom["ipp"]
    dx = geom["dx"]
    dy = geom["dy"]

    # ImagePositionPatient 是左上角首像素中心；变换中心对齐到切片几何中心
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    center_world = ipp + row_cos * dx * cx + col_cos * dy * cy

    mat = np.zeros((3, 4), dtype=np.float32)
    mat[:, 0] = row_cos.astype(np.float32)
    mat[:, 1] = col_cos.astype(np.float32)
    mat[:, 2] = normal.astype(np.float32)
    mat[:, 3] = center_world.astype(np.float32)
    return mat


def _build_stack_from_dcm_files(
    dcm_files: Sequence[str],
    stack_name: str,
    frame_idx: int,
    timestamp: float,
    device: torch.device,
) -> Tuple[Optional[Stack], List[Dict[str, Any]]]:
    if not dcm_files:
        return None, []

    records: List[Dict[str, Any]] = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            geom = _read_dicom_geometry(ds)
            h, w = img.shape
            mat = _slice_transform_matrix(geom, h, w)
            pos = float(np.dot(geom["ipp"], geom["normal"]))
            records.append(
                {
                    "path": f,
                    "img": img,
                    "mask": generate_background_mask(img),
                    "mat": mat,
                    "position": pos,
                    "dx": geom["dx"],
                    "dy": geom["dy"],
                    "thickness": geom["thickness"],
                    "timestamp": timestamp,
                    "frame_idx": frame_idx,
                    "stack_name": stack_name,
                }
            )
        except Exception as exc:
            logging.warning("Skip DICOM %s due to parsing error: %s", f, exc)

    if not records:
        return None, []

    records = sorted(records, key=lambda r: r["position"])

    # 以第一张切片的分辨率作为该 stack 的统一定义
    dx = float(records[0]["dx"])
    dy = float(records[0]["dy"])
    thickness = float(records[0]["thickness"])
    if len(records) > 1:
        positions = np.array([r["position"] for r in records], dtype=np.float64)
        gap = float(np.median(np.abs(np.diff(positions))))
        if gap <= 0:
            gap = thickness
    else:
        gap = thickness

    slices = torch.from_numpy(
        np.stack([r["img"] for r in records], axis=0)
    ).to(device=device, dtype=torch.float32)
    masks = torch.from_numpy(
        np.stack([r["mask"] for r in records], axis=0)
    ).to(device=device, dtype=torch.bool)
    mats = torch.from_numpy(
        np.stack([r["mat"] for r in records], axis=0)
    ).to(device=device, dtype=torch.float32)

    stack = Stack(
        slices=slices.unsqueeze(1),
        mask=masks.unsqueeze(1),
        transformation=RigidTransform(mats, trans_first=True),
        resolution_x=dx,
        resolution_y=dy,
        thickness=thickness,
        gap=gap,
        name=f"{stack_name}_frame_{frame_idx:02d}",
    )

    return stack, records


def _registration_flags(method: str) -> Tuple[bool, bool, bool]:
    if method == "svort":
        return True, True, False
    if method == "svort-stack":
        return True, True, True
    if method == "svort-only":
        return True, False, False
    if method == "stack":
        return False, True, False
    if method == "none":
        return False, False, False
    raise ValueError(f"Unknown registration method: {method}")


def _run_source_registration(
    stacks: List[Stack],
    registration: str,
    device: torch.device,
    svort_version: str,
    scanner_space: bool,
):
    svort, vvr, force_vvr = _registration_flags(registration)
    return svort_predict(
        stacks,
        device=device,
        svort_version=svort_version,
        svort=svort,
        vvr=vvr,
        force_vvr=force_vvr,
        force_scanner=scanner_space,
    )


def _slice_to_numpy(slice_obj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = slice_obj.image.squeeze(0).detach().cpu().numpy().astype(np.float32)
    mask = slice_obj.mask.squeeze(0).detach().cpu().numpy().astype(bool)
    affine = transformation2affine(
        slice_obj.image,
        slice_obj.transformation,
        float(slice_obj.resolution_x),
        float(slice_obj.resolution_y),
        float(slice_obj.resolution_z),
    ).astype(np.float32)
    return img, affine, mask


def load_cine_dicom_dataset(
    base_dir: str, stack_names: Optional[Sequence[str]] = None
):
    """
    直接读取 DICOM（不做注册），返回 4D 训练所需列表
    """
    stacks = _default_stacks(stack_names)

    images_list: List[np.ndarray] = []
    affines_list: List[np.ndarray] = []
    timestamps_list: List[float] = []
    masks_list: List[np.ndarray] = []

    logging.info("Scanning DICOM directory: %s", base_dir)

    for stack_name in stacks:
        stack_dir = os.path.join(base_dir, stack_name)
        if not os.path.exists(stack_dir):
            logging.warning("Stack directory not found, skipping: %s", stack_dir)
            continue

        dcm_files = glob.glob(os.path.join(stack_dir, "**", "*.dcm"), recursive=True)
        logging.info("Found %d DICOM files in %s.", len(dcm_files), stack_name)

        trigger_times: List[float] = []
        valid_files: List[str] = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                if "TriggerTime" in ds:
                    trigger_times.append(float(ds.TriggerTime))
                    valid_files.append(f)
            except Exception:
                continue

        if not valid_files:
            continue

        unique_phases = sorted({round(t, 1) for t in trigger_times})
        num_phases = len(unique_phases)
        logging.info(
            "Detected %d distinct cardiac phases in %s.", num_phases, stack_name
        )

        for f in valid_files:
            ds_full = pydicom.dcmread(f)
            img = ds_full.pixel_array.astype(np.float32)
            affine = compute_affine_from_dicom(ds_full).astype(np.float32)
            current_phase = round(float(ds_full.TriggerTime), 1)
            phase_idx = unique_phases.index(current_phase)
            t_val = phase_idx / num_phases
            mask = generate_background_mask(img)

            images_list.append(img)
            affines_list.append(affine)
            timestamps_list.append(float(t_val))
            masks_list.append(mask)

    logging.info("Dataset fully loaded! Total 2D slices: %d", len(images_list))
    return images_list, affines_list, timestamps_list, masks_list


def load_cine_dicom_dataset_with_source_registration(
    base_dir: str,
    stack_names: Optional[Sequence[str]] = None,
    registration: str = "stack",
    device: str = "cuda",
    svort_version: str = "v2",
    scanner_space: bool = False,
):
    """
    按 frame 组装 stacks，并调用源码 registration（svort_predict）后再导出 2D 切片。
    """
    stacks = _default_stacks(stack_names)
    device_t = torch.device(device)

    frame_map: Dict[str, Dict[int, str]] = {}
    for stack_name in stacks:
        frame_root = os.path.join(base_dir, f"{stack_name}_frames")
        if not os.path.isdir(frame_root):
            logging.warning(
                "Frame directory %s not found. Fallback to direct DICOM loading.",
                frame_root,
            )
            return load_cine_dicom_dataset(base_dir, stack_names=stacks)
        idx_to_dir: Dict[int, str] = {}
        for name in os.listdir(frame_root):
            full = os.path.join(frame_root, name)
            if not os.path.isdir(full):
                continue
            idx = _parse_frame_idx(name)
            if idx is None:
                continue
            idx_to_dir[idx] = full
        frame_map[stack_name] = idx_to_dir

    all_frame_idx = sorted(
        {idx for idx_to_dir in frame_map.values() for idx in idx_to_dir.keys()}
    )
    if not all_frame_idx:
        logging.warning("No frame folders found. Fallback to direct DICOM loading.")
        return load_cine_dicom_dataset(base_dir, stack_names=stacks)

    images_list: List[np.ndarray] = []
    affines_list: List[np.ndarray] = []
    timestamps_list: List[float] = []
    masks_list: List[np.ndarray] = []

    n_frames = len(all_frame_idx)
    logging.info(
        "Using source registration=%s on %d frames, stacks=%s",
        registration,
        n_frames,
        stacks,
    )

    for order_idx, frame_idx in enumerate(all_frame_idx):
        timestamp = float(order_idx / n_frames)
        frame_stacks: List[Stack] = []
        frame_meta: List[List[Dict[str, Any]]] = []

        for stack_name in stacks:
            frame_dir = frame_map.get(stack_name, {}).get(frame_idx)
            if frame_dir is None:
                continue
            dcm_files = sorted(glob.glob(os.path.join(frame_dir, "*.dcm")))
            stack_obj, records = _build_stack_from_dcm_files(
                dcm_files=dcm_files,
                stack_name=stack_name,
                frame_idx=frame_idx,
                timestamp=timestamp,
                device=device_t,
            )
            if stack_obj is None:
                continue
            frame_stacks.append(stack_obj)
            frame_meta.append(records)

        if not frame_stacks:
            continue

        if registration != "none":
            ordered_meta: List[Dict[str, Any]] = []
            for stack_obj, records in zip(frame_stacks, frame_meta):
                idx_nonempty = (
                    stack_obj.mask.flatten(1).any(1).detach().cpu().tolist()
                )
                for keep, rec in zip(idx_nonempty, records):
                    if keep:
                        ordered_meta.append(rec)

            registered_slices = _run_source_registration(
                stacks=frame_stacks,
                registration=registration,
                device=device_t,
                svort_version=svort_version,
                scanner_space=scanner_space,
            )
            if len(registered_slices) != len(ordered_meta):
                logging.warning(
                    "Frame %02d: registration slice count mismatch (%d vs %d). "
                    "Use shortest length.",
                    frame_idx,
                    len(registered_slices),
                    len(ordered_meta),
                )
            keep_n = min(len(registered_slices), len(ordered_meta))
            for i in range(keep_n):
                img, affine, mask = _slice_to_numpy(registered_slices[i])
                images_list.append(img)
                affines_list.append(affine)
                timestamps_list.append(float(ordered_meta[i]["timestamp"]))
                masks_list.append(mask)
        else:
            for stack_obj, records in zip(frame_stacks, frame_meta):
                idx_nonempty = (
                    stack_obj.mask.flatten(1).any(1).detach().cpu().tolist()
                )
                for i, (keep, rec) in enumerate(zip(idx_nonempty, records)):
                    if not keep:
                        continue
                    sl = stack_obj[i]
                    img, affine, mask = _slice_to_numpy(sl)
                    images_list.append(img)
                    affines_list.append(affine)
                    timestamps_list.append(float(rec["timestamp"]))
                    masks_list.append(mask)

    logging.info(
        "Dataset loaded with source registration. Total 2D slices: %d",
        len(images_list),
    )
    return images_list, affines_list, timestamps_list, masks_list
