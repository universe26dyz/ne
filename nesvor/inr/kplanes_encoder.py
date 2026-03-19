import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn
from torch.nn import functional as F



def normalize_aabb(pts, aabb):
    """将点坐标从AABB（轴向包围盒）空间归一化到[-1, 1]范围

    Args:
        pts: 输入点坐标，形状为[..., 3]
        aabb: 轴向包围盒，形状为[2, 3]，aabb[0]为最小值，aabb[1]为最大值
    
    Returns:
        归一化到[-1, 1]范围内的点坐标
    """
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    """原封不动保留 K-Planes 原生的 Grid 初始化"""
    """初始化K-Plane网格参数

    Args:
        grid_nd: 网格的维度（通常是2，表示2D平面）
        in_dim: 输入坐标的维度（3表示空间，4表示时空）
        out_dim: 每个平面输出的特征维度
        reso: 每个维度的分辨率
        a, b: 均匀分布初始化参数
    
    Returns:
        包含所有平面参数的ParameterList
    """
    assert in_dim == len(reso), "分辨率数量必须与输入维度相同"
    has_time_planes = in_dim == 4  # 检查是否为时空模型
    assert grid_nd <= in_dim
    
    # 生成所有可能的坐标组合（用于创建不同的平面）
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    
    for ci, coo_comb in enumerate(coo_combs):
        # 创建网格参数：形状为[1, out_dim, reso1, reso2, ...]
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        
        # 如果是时间平面且包含时间维度（索引3），初始化为1
        if has_time_planes and 3 in coo_comb:
            nn.init.ones_(new_grid_coef)
        else:
            # 空间平面使用均匀分布初始化
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    grid_dim = coords.shape[-1]

    if grid.dim() == grid_dim + 1:
        # no batch dimension present, need to add it
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]
        coords,  # [B, 1, ..., n, grid_dim]
        align_corners=align_corners,
        mode='bilinear', padding_mode='border')
    interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp


class KPlanesFeatureEncoder(nn.Module):
    def __init__(
        self,
        aabb, # [2, 4] 包含 [min_x, min_y, min_z, min_t] 和 max
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]] = None,
        v3d_scale_indices: List[int] = [0],  # 【新增】指定哪些尺度用于计算 V_3D
        fusion_mode: str = "mul",
    ) -> None:
        """
        纯粹的 K-Planes 特征编码与分流器 (无 MLP)
        """
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.grid_config = grid_config
        self.multiscale_res_multipliers = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.v3d_scale_indices = v3d_scale_indices
        self.fusion_mode = fusion_mode

        if self.fusion_mode not in ("mul", "grouped_mlp"):
            raise ValueError(
                f"Unsupported fusion_mode={self.fusion_mode}. Expected one of ['mul', 'grouped_mlp']."
            )

        self.grids = nn.ModuleList()
        self.spatial_fusion_mlps = nn.ModuleList()
        self.temporal_fusion_mlps = nn.ModuleList()
        self.st_fusion_mlps = nn.ModuleList()
        self.register_buffer(
            "active_scale_mask",
            torch.ones(len(self.multiscale_res_multipliers), dtype=torch.bool),
        )
        
        # 记录输出的特征维度，方便外部 MLP 自动对齐
        self.v4d_feature_dim = 0
        self.v3d_feature_dim = 0

        for scale_id, res in enumerate(self.multiscale_res_multipliers):
            config = self.grid_config[0].copy()
            # 空间分辨率翻倍，时间分辨率保持
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],      # 必须是 2 (代表 2D 平面)
                in_dim=config["input_coordinate_dim"],  # 必须是 4 (x,y,z,t)
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
                a=0.8,  # <--- 新增：提高初始化的下限
                b=1.2  # <--- 新增：提高初始化的上限
            )
            self.grids.append(gp)
            
            # 计算输出维度
            scale_out_dim = gp[-1].shape[1]
            if self.fusion_mode == "grouped_mlp":
                self.spatial_fusion_mlps.append(
                    nn.Sequential(
                        nn.Linear(3 * scale_out_dim, scale_out_dim),
                        nn.SiLU(),
                        nn.Linear(scale_out_dim, scale_out_dim),
                    )
                )
                self.temporal_fusion_mlps.append(
                    nn.Sequential(
                        nn.Linear(3 * scale_out_dim, scale_out_dim),
                        nn.SiLU(),
                        nn.Linear(scale_out_dim, scale_out_dim),
                    )
                )
                self.st_fusion_mlps.append(
                    nn.Sequential(
                        nn.Linear(2 * scale_out_dim, scale_out_dim),
                        nn.SiLU(),
                        nn.Linear(scale_out_dim, scale_out_dim),
                    )
                )
            if self.concat_features:
                self.v4d_feature_dim += scale_out_dim
                if scale_id in self.v3d_scale_indices:
                    self.v3d_feature_dim += scale_out_dim
            else:
                self.v4d_feature_dim = scale_out_dim
                if scale_id in self.v3d_scale_indices:
                    self.v3d_feature_dim = scale_out_dim

        log.info(
            f"K-Planes Encoder Init: fusion_mode={self.fusion_mode}, "
            f"V_4D dim={self.v4d_feature_dim}, V_3D dim={self.v3d_feature_dim} "
            f"(Using scales {self.v3d_scale_indices})"
        )

    def set_active_scale_mask(self, active_scale_mask) -> None:
        mask = torch.as_tensor(
            active_scale_mask, dtype=torch.bool, device=self.active_scale_mask.device
        )
        if mask.shape != self.active_scale_mask.shape:
            raise ValueError(
                f"active_scale_mask shape mismatch: got {tuple(mask.shape)}, "
                f"expected {tuple(self.active_scale_mask.shape)}"
            )
        self.active_scale_mask.copy_(mask)

    def get_active_scale_ids(self) -> List[int]:
        return [
            scale_id
            for scale_id, is_active in enumerate(self.active_scale_mask.tolist())
            if is_active
        ]

    def get_active_scale_values(self) -> List[int]:
        return [
            self.multiscale_res_multipliers[scale_id]
            for scale_id in self.get_active_scale_ids()
        ]

    def is_scale_active(self, scale_id: int) -> bool:
        return bool(self.active_scale_mask[scale_id].item())

    def forward(self, xyzt: torch.Tensor):
        # 1. 坐标归一化到 [-1, 1]
        pts_normalized = normalize_aabb(xyzt, self.aabb).view(-1, 4)
        n_pts = pts_normalized.shape[0]

        multi_scale_v3d = [] if self.concat_features else 0.
        multi_scale_v4d = [] if self.concat_features else 0.

        # 4D 坐标两两组合共有 6 个平面。itertools.combinations(range(4), 2) 的顺序是固定且明确的：
        # 0: (0, 1) -> xy
        # 1: (0, 2) -> xz
        # 2: (0, 3) -> xt
        # 3: (1, 2) -> yz
        # 4: (1, 3) -> yt
        # 5: (2, 3) -> zt
        
        spatial_idx = [(0, 1, 0), (0, 2, 1), (1, 2, 3)]
        temporal_idx = [(0, 3, 2), (1, 3, 4), (2, 3, 5)]

        # 2. 遍历多尺度网格进行插值和物理分流
        for scale_id, grid in enumerate(self.grids):
            feature_dim = grid[-1].shape[1]
            if not self.is_scale_active(scale_id):
                zero_feat = pts_normalized.new_zeros((n_pts, feature_dim))
                if self.concat_features:
                    multi_scale_v4d.append(zero_feat)
                else:
                    multi_scale_v4d = multi_scale_v4d + zero_feat

                if scale_id in self.v3d_scale_indices:
                    if self.concat_features:
                        multi_scale_v3d.append(zero_feat)
                    else:
                        multi_scale_v3d = multi_scale_v3d + zero_feat
                continue

            plane_feats = {}
            for c1, c2, g_idx in spatial_idx + temporal_idx:
                plane_feats[g_idx] = grid_sample_wrapper(
                    grid[g_idx], pts_normalized[..., [c1, c2]]
                ).view(-1, feature_dim)

            f_xy = plane_feats[0]
            f_xz = plane_feats[1]
            f_xt = plane_feats[2]
            f_yz = plane_feats[3]
            f_yt = plane_feats[4]
            f_zt = plane_feats[5]

            if self.fusion_mode == "mul":
                v3d_scale = f_xy * f_xz * f_yz
                v4d_scale = v3d_scale * f_xt * f_yt * f_zt
            else:
                s_in = torch.cat([f_xy, f_xz, f_yz], dim=-1)
                v3d_scale = self.spatial_fusion_mlps[scale_id](s_in)

                t_in = torch.cat([f_xt, f_yt, f_zt], dim=-1)
                t_scale = self.temporal_fusion_mlps[scale_id](t_in)

                st_in = torch.cat([v3d_scale, t_scale], dim=-1)
                v4d_scale = self.st_fusion_mlps[scale_id](st_in)

            # 2.3 分流收集 (核心逻辑)
            # V_4D 收集所有尺度
            if self.concat_features:
                multi_scale_v4d.append(v4d_scale)
            else:
                multi_scale_v4d = multi_scale_v4d + v4d_scale

            # V_3D 仅收集用户指定的尺度 (例如只收集低频尺度 0)
            if scale_id in self.v3d_scale_indices:
                if self.concat_features:
                    multi_scale_v3d.append(v3d_scale)
                else:
                    multi_scale_v3d = multi_scale_v3d + v3d_scale

        # 3. 拼接输出
        if self.concat_features:
            v_3d = torch.cat(multi_scale_v3d, dim=-1)
            v_4d = torch.cat(multi_scale_v4d, dim=-1)
        else:
            v_3d = multi_scale_v3d
            v_4d = multi_scale_v4d

        return v_3d, v_4d
