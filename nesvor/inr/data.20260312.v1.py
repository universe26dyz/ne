from typing import Dict, List
import torch
from torch.utils.data import Dataset
import numpy as np
from ..utils import gaussian_blur
# 确保这里导入了 transform_points
from ..transform import RigidTransform, transform_points 
from ..image import Volume, Slice

#######已封存的v1版本的。
class PointDataset(object):

    def __init__(self, slices: List[Slice]) -> None:
        self.mask_threshold = 1  # args.mask_threshold

        xyz_all = []
        v_all = []
        slice_idx_all = []
        transformation_all = []
        resolution_all = []
        
        # --- 新增：用于计算固定 Bounding Box 的列表 ---
        corners_all = [] 
        # -------------------------------------------

        for i, slice in enumerate(slices):
            # 1. 收集训练用的采样点 (和原来一样)
            xyz = slice.xyz_masked_untransformed
            v = slice.v_masked
            slice_idx = torch.full(v.shape, i, device=v.device)
            xyz_all.append(xyz)
            v_all.append(v)
            slice_idx_all.append(slice_idx)
            transformation_all.append(slice.transformation)
            resolution_all.append(slice.resolution_xyz)
            
            # 2. --- 新增：计算该切片的物理角点 ---
            # NeSVoR 的 Slice 对象通常有 .shape 属性
            if hasattr(slice, 'shape'):
                H, W = slice.shape[-2], slice.shape[-1]
            else:
                # 回退逻辑
                H, W = 256, 256 
            
            # 定义切片在局部坐标系的 4 个角点 (z=0)
            res = slice.resolution_xyz
            # 局部坐标 (x, y, z)
            corners = torch.tensor([
                [0.0, 0.0, 0.0],
                [0.0, float(H-1), 0.0],
                [float(W-1), 0.0, 0.0],
                [float(W-1), float(H-1), 0.0]
            ], device=v.device)
            
            # 转换到物理毫米单位
            corners = corners * res
            
            # --- 修正点：使用独立函数 transform_points ---
            # 原错误代码: corners_world = slice.transformation.transform_points(corners)
            corners_world = transform_points(slice.transformation, corners)
            # ----------------------------------------
            
            corners_all.append(corners_world)

        self.xyz = torch.cat(xyz_all)
        self.v = torch.cat(v_all)
        self.slice_idx = torch.cat(slice_idx_all)
        self.transformation = RigidTransform.cat(transformation_all)
        self.resolution = torch.stack(resolution_all, 0)
        self.count = self.v.shape[0]
        self.epoch = 0
        
        # --- 新增：预计算全局固定的 Bounding Box ---
        # 无论 mask 怎么变，这个 box 由图像头文件决定，绝对固定
        all_corners = torch.cat(corners_all, dim=0)
        
        # 为了安全，稍微外扩一点点 resolution
        max_r = self.resolution.max()
        self.fixed_xyz_min = all_corners.amin(0) - max_r
        self.fixed_xyz_max = all_corners.amax(0) + max_r
        # ---------------------------------------

    @property
    def bounding_box(self) -> torch.Tensor:
        # --- 修改：直接返回预计算的固定 Box ---
        return torch.stack([self.fixed_xyz_min, self.fixed_xyz_max], 0)

    @property
    def mean(self) -> float:
        q1, q2 = torch.quantile(
            self.v if self.v.numel() < 256 * 256 * 256 else self.v[: 256 * 256 * 256],
            torch.tensor([0.1, 0.9], dtype=self.v.dtype, device=self.v.device),
        )
        return self.v[torch.logical_and(self.v > q1, self.v < q2)].mean().item()

    def get_batch(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        if self.count + batch_size > self.xyz.shape[0]:  # new epoch, shuffle data
            self.count = 0
            self.epoch += 1
            idx = torch.randperm(self.xyz.shape[0], device=device)
            self.xyz = self.xyz[idx]
            self.v = self.v[idx]
            self.slice_idx = self.slice_idx[idx]
        # fetch a batch of data
        batch = {
            "xyz": self.xyz[self.count : self.count + batch_size],
            "v": self.v[self.count : self.count + batch_size],
            "slice_idx": self.slice_idx[self.count : self.count + batch_size],
        }
        self.count += batch_size
        return batch

    @property
    def xyz_transformed(self) -> torch.Tensor:
        return transform_points(self.transformation[self.slice_idx], self.xyz)

    @property
    def mask(self) -> Volume:
        with torch.no_grad():
            resolution_min = self.resolution.min()
            resolution_max = self.resolution.max()
            xyz = self.xyz_transformed
            xyz_min = xyz.amin(0) - resolution_max * 10
            xyz_max = xyz.amax(0) + resolution_max * 10
            shape_xyz = ((xyz_max - xyz_min) / resolution_min).ceil().long()
            shape = (int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]))
            kji = ((xyz - xyz_min) / resolution_min).round().long()

            mask = torch.bincount(
                kji[..., 0]
                + shape[2] * kji[..., 1]
                + shape[2] * shape[1] * kji[..., 2],
                minlength=shape[0] * shape[1] * shape[2],
            )
            mask = mask.view((1, 1) + shape).float()
            mask_threshold = (
                self.mask_threshold
                * resolution_min**3
                / self.resolution.log().mean().exp() ** 3
            )
            mask_threshold *= mask.sum() / (mask > 0).sum()
            assert len(mask.shape) == 5
            mask = (
                gaussian_blur(mask, (resolution_max / resolution_min).item(), 3)
                > mask_threshold
            )[0, 0]

            xyz_c = xyz_min + (shape_xyz - 1) / 2 * resolution_min
            return Volume(
                mask.float(),
                mask,
                RigidTransform(torch.cat([0 * xyz_c, xyz_c])[None], True),
                resolution_min,
                resolution_min,
                resolution_min,
            )
        
class DynamicMRIDataset(Dataset):
    def __init__(self, 
                 images_list, 
                 affines_list, 
                 timestamps_list, 
                 nav_offsets_list=None, 
                 masks_list=None,
                 psf_resolution=(1.0, 1.0, 5.0)): # (dx, dy, dz) 层厚 dz 通常远大于面内 dx dy
        """
        4D MRI (Cine/Real-time) 数据集
        :param images_list: list of [H, W] 张量或 numpy 数组 (单张 2D 切片)
        :param affines_list: list of [4, 4] 仿射矩阵 (从像素网格到物理空间的映射)
        :param timestamps_list: list of floats, 心跳相位 t \in [0, 1)
        :param nav_offsets_list: list of floats, 膈肌导航偏移量 (可选)
        :param masks_list: list of [H, W] 布尔张量, 感兴趣区域掩膜 (过滤背景加速训练)
        :param psf_resolution: 物理分辨率 (用于后续各向异性高斯采样)
        """
        super().__init__()
        
        self.psf_resolution = torch.tensor(psf_resolution, dtype=torch.float32)
        
        all_xyzt = []
        all_v = []
        all_slice_idx = []
        all_slice_R = [] # 保存切片的旋转矩阵，用于后续各向异性 PSF 扰动
        
        N_slices = len(images_list)
        print(f"[{self.__class__.__name__}] Parsing {N_slices} slices...")
        
        for idx in range(N_slices):
            img = torch.as_tensor(images_list[idx], dtype=torch.float32)
            affine = torch.as_tensor(affines_list[idx], dtype=torch.float32)
            t_val = float(timestamps_list[idx])
            
            H, W = img.shape
            
            # 1. 获取有效像素的 Mask
            if masks_list is not None:
                mask = torch.as_tensor(masks_list[idx], dtype=torch.bool)
            else:
                mask = img > 0  # 简单的背景过滤
                
            valid_coords = torch.where(mask)  # (i_indices, j_indices)
            i_coords = valid_coords[0].float()
            j_coords = valid_coords[1].float()
            
            # 2. 构造图像平面网格局部坐标 [V, 3] -> (i, j, 0)
            V = i_coords.shape[0]
            if V == 0:
                continue
            local_coords = torch.stack([i_coords, j_coords, torch.zeros(V)], dim=-1)
            
            # 3. 仿射变换: 提取旋转矩阵 R 和平移向量 T
            # R = affine[:3, :3]  # [3, 3]
            # T = affine[:3, 3]   # [3]
            
            # # 像素坐标 -> 物理空间绝对坐标 [V, 3]
            # xyz = torch.matmul(local_coords, R.T) + T
                        # 建议改为
            R_world = affine[:3, :3]   # 含spacing，用于world坐标
            T = affine[:3, 3]
            xyz = torch.matmul(local_coords, R_world.T) + T

            # 仅用于PSF方向旋转：构造单位正交基
            c0 = R_world[:, 0]
            c1 = R_world[:, 1]

            e0 = c0 / (torch.norm(c0) + 1e-8)
            e1_raw = c1 / (torch.norm(c1) + 1e-8)
            e2 = torch.cross(e0, e1_raw, dim=0)
            e2 = e2 / (torch.norm(e2) + 1e-8)

            # 重新正交化第二列，避免数值漂移
            e1 = torch.cross(e2, e0, dim=0)
            e1 = e1 / (torch.norm(e1) + 1e-8)
                        
            # 4. (可选) 呼吸导航 NAV 修正 Z 轴
            if nav_offsets_list is not None:
                nav_offset = nav_offsets_list[idx]
                # 假设 NAV 偏移作用于物理 Z 轴
                xyz[:, 2] -= nav_offset
                
            # 5. 组装时空 4D 坐标: 把时间 t 拼接到最后 -> [V, 4]
            # 【符合规则1：时间 t 在此处拼接，网络内部先拆开处理 xyz 再拼回】
            t_tensor = torch.full((V, 1), t_val, dtype=torch.float32)
            xyzt = torch.cat([xyz, t_tensor], dim=-1)  # [V, 4]
            
            # 6. 提取对应的真实像素值 v
            v = img[valid_coords].unsqueeze(-1)  # [V, 1]
            
            # 7. 记录切片编号 (用于 NeSVoR 切片独立变量 axisangle/DeformNet)
            slice_idx_tensor = torch.full((V, 1), idx, dtype=torch.long) # [V, 1]
            
            # 8. 记录旋转矩阵并扩展 (PSF 需要依靠这个 R 来确定哪个方向是层厚)
            # slice_R_tensor = R.unsqueeze(0).expand(V, 3, 3) # [V, 3, 3]
            R_unit = torch.stack([e0, e1, e2], dim=-1)   # [3,3], unit orthogonal
            slice_R_tensor = R_unit.unsqueeze(0).expand(V, 3, 3)

            all_xyzt.append(xyzt)
            all_v.append(v)
            all_slice_idx.append(slice_idx_tensor)
            all_slice_R.append(slice_R_tensor)
            
        # 拼接所有切片的有效像素，打平成用于 NeRF 训练的巨大 Ray 集合
        self.xyzt = torch.cat(all_xyzt, dim=0)             # [Total_Rays, 4]
        self.v = torch.cat(all_v, dim=0)                   # [Total_Rays, 1]
        self.slice_idx = torch.cat(all_slice_idx, dim=0)   # [Total_Rays, 1]
        self.slice_R = torch.cat(all_slice_R, dim=0)       # [Total_Rays, 3, 3]
        
        self.total_samples = self.xyzt.shape[0]
        print(f"[{self.__class__.__name__}] Generated {self.total_samples} valid 4D rays.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        DataLoader 将返回形如 [Batch, ...] 的字典
        """
        return {
            "xyzt": self.xyzt[idx],               # [4]   (x, y, z, t)
            "v": self.v[idx],                     # [1]   真实强度
            "slice_idx": self.slice_idx[idx],     # [1]   切片索引，非常关键！
            "slice_R": self.slice_R[idx],         # [3, 3] 局部到物理的旋转矩阵
        }