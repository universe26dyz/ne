from typing import List, Union, Optional
import os
import torch
import torch.nn as nn
from ..transform import transform_points, RigidTransform
from ..image import Slice, Volume, load_volume, load_mask
from .models import INR
from ..utils import resolution2sigma, meshgrid, PathType


def override_sample_mask(
    mask: Volume,
    new_mask: Union[PathType, None, Volume] = None,
    new_resolution: Optional[float] = None,
    new_orientation: Union[PathType, None, Volume, RigidTransform] = None,
) -> Volume:
    if new_mask is not None:
        if isinstance(new_mask, Volume):
            mask = new_mask
        elif isinstance(new_mask, (str, os.PathLike)):
            mask = load_mask(new_mask, device=mask.device)
        else:
            raise TypeError("unknwon type for mask")
    transformation = None
    if new_orientation is not None:
        if isinstance(new_orientation, Volume):
            transformation = new_orientation.transformation
        elif isinstance(new_orientation, RigidTransform):
            transformation = new_orientation
        elif isinstance(new_orientation, (str, os.PathLike)):
            transformation = load_volume(
                new_orientation,
                device=mask.device,
            ).transformation
    if transformation or new_resolution:
        mask = mask.resample(new_resolution, transformation)
    return mask


def sample_volume(
    model: INR,
    mask: Volume,
    psf_resolution: float,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> Volume:
    model.eval()
    img = mask.clone()
    img.image[img.mask] = sample_points(
        model,
        img.xyz_masked,
        psf_resolution,
        batch_size,
        n_samples,
    )
    return img


def sample_points(
    model: INR,
    xyz: torch.Tensor,
    resolution: float = 0,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> torch.Tensor:
    shape = xyz.shape[:-1]
    xyz = xyz.view(-1, 3)
    v = torch.empty(xyz.shape[0], dtype=torch.float32, device=xyz.device)
    with torch.no_grad():
        for i in range(0, xyz.shape[0], batch_size):
            xyz_batch = xyz[i : i + batch_size]
            xyz_batch = model.sample_batch(
                xyz_batch,
                None,
                resolution2sigma(resolution, isotropic=True),
                0 if resolution <= 0 else n_samples,
            )
            v_b = model(xyz_batch).mean(-1)
            v[i : i + batch_size] = v_b
    return v.view(shape)


def sample_slice(
    model: INR,
    slice: Slice,
    mask: Volume,
    output_psf_factor: float = 1.0,
    n_samples: int = 128,
) -> Slice:
    slice_sampled = slice.clone(zero=True)
    xyz = meshgrid(slice_sampled.shape_xyz, slice_sampled.resolution_xyz).view(-1, 3)
    m = mask.sample_points(transform_points(slice_sampled.transformation, xyz)) > 0
    if m.any():
        xyz_masked = model.sample_batch(
            xyz[m],
            slice_sampled.transformation,
            resolution2sigma(
                slice_sampled.resolution_xyz * output_psf_factor, isotropic=False
            ),
            0 if output_psf_factor <= 0 else n_samples,
        )
        v = model(xyz_masked).mean(-1)
        slice_sampled.mask = m.view(slice_sampled.mask.shape)
        slice_sampled.image[slice_sampled.mask] = v.to(slice_sampled.image.dtype)
    return slice_sampled


def sample_slices(
    model: INR,
    slices: List[Slice],
    mask: Volume,
    output_psf_factor: float = 1.0,
    n_samples: int = 128,
) -> List[Slice]:
    model.eval()
    with torch.no_grad():
        slices_sampled = []
        for i, slice in enumerate(slices):
            slices_sampled.append(
                sample_slice(model, slice, mask, output_psf_factor, n_samples)
            )
    return slices_sampled


def sample_4d_with_psf(xyzt, slice_idx, slice_R, axisangle, psf_sigma, n_samples):
    """
    针对 4D 时空数据的各向异性 PSF 采样器
    
    :param xyzt:        [Batch, 4] DataLoader传来的物理坐标与时间 (x,y,z,t)
    :param slice_idx:   [Batch, 1] 切片编号
    :param slice_R:     [Batch, 3, 3] 当前切片的局部到世界旋转矩阵
    :param axisangle:   [N_slices, 6] 模型可学习的位姿参数 (前3平移，后3旋转)
    :param psf_sigma:   [3] 或 [1, 3] 物理分辨率标准差，形如 (dx, dy, dz)，dz 通常远大于 dx, dy
    :param n_samples:   int, 每条射线(每个像素)采样的高斯点数量 S
    
    :return: 
        xyzt_sampled: [Batch, n_samples, 4] 最终送入 K-Planes 的合法四维张量
    """
    B = xyzt.shape[0]
    S = n_samples
    
    # ==========================================
    # 1. 严格时空剥离 (保护 t 不受任何物理空间变换的污染)
    # ==========================================
    xyz = xyzt[:, :3]  # [B, 3]
    t = xyzt[:, 3:]    # [B, 1]
    
    # ==========================================
    # 2. 切片位姿优化 (Pose T 变换)
    # ==========================================
    # 取出当前 batch 对应切片的可学习位姿参数
    # axisangle 包含了所有切片的位姿，形状为 [Total_Slices, 6]
    batch_axisangle = axisangle[slice_idx.squeeze(-1)] # [B, 6]
    
    # 使用 NeSVoR 的 RigidTransform 应用可学习的刚体变换
    transform = RigidTransform(batch_axisangle)
    xyz_transformed = transform(xyz) # [B, 3] 经过网络优化的物理中心点
    
    # 同样的，由于切片位姿发生了旋转，理论上局部坐标系的旋转矩阵也要附带这个微小旋转
    slice_R_transformed = transform.matrix[:, :3, :3] @ slice_R  # [B, 3, 3] (可选：如果切片旋转角度较大，建议加上这一步)
    # slice_R_transformed = slice_R # 通常运动微小，直接使用原仿射矩阵的R也可
    
    # ==========================================
    # 3. 各向异性 3D 高斯采样 (PSF)
    # ==========================================
    # 3.1 在切片“局部”坐标系下生成高斯噪声 [B, S, 3]
    # psf_sigma 设置为例如 tensor([1.0, 1.0, 5.0])，使得层厚方向(z)的扰动极大
    if not isinstance(psf_sigma, torch.Tensor):
        psf_sigma = torch.tensor(psf_sigma, dtype=torch.float32, device=xyz.device)
    
    noise_local = torch.randn(B, S, 3, device=xyz.device) * psf_sigma # [B, S, 3]
    
    # 3.2 将局部噪声旋转到世界物理空间 [B, S, 3]
    # 利用爱因斯坦求和约定进行批量矩阵乘法：slice_R_transformed[B, i, j] * noise_local[B, S, j] -> noise_world[B, S, i]
    noise_world = torch.einsum('bij,bsj->bsi', slice_R_transformed, noise_local)
    
    # 3.3 噪声叠加到中心点
    xyz_expanded = xyz_transformed.unsqueeze(1) # [B, 1, 3]
    xyz_sampled = xyz_expanded + noise_world    # [B, S, 3]
    
    # ==========================================
    # 4. 时空重组
    # ==========================================
    # 扩展时间维度以匹配采样点数量
    t_expanded = t.unsqueeze(1).expand(B, S, 1) # [B, S, 1]
    
    # 拼合！形成最终合法的 4D 坐标
    xyzt_sampled = torch.cat([xyz_sampled, t_expanded], dim=-1) # [B, S, 4]
    
    # 折叠前两个维度，方便后续送入全连接层和 K-Planes [B*S, 4]
    xyzt_flat = xyzt_sampled.view(B * S, 4)
    
    return xyzt_flat