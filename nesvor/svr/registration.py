import types
from typing import Dict, Any, Tuple, Callable, Union, cast, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..transform import RigidTransform, axisangle2mat, mat_update_resolution
from ..utils import ncc_loss, gaussian_blur, meshgrid, resample
from ..slice_acquisition import slice_acquisition
from ..image import Volume, Stack


class Registration(nn.Module):
    def __init__(
        self,
        num_levels: int = 3,
        num_steps: int = 4,
        step_size: float = 2,
        max_iter: int = 20,
        optimizer: Optional[Dict[str, Any]] = None,
        loss: Optional[Union[Dict[str, Any], Callable]] = None,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.current_level = self.num_levels - 1
        self.num_steps = [num_steps] * self.num_levels
        self.step_sizes = [step_size * 2**level for level in range(num_levels)]
        self.max_iter = max_iter
        self.auto_grad = False
        self._degree2rad = torch.tensor(
            [np.pi / 180, np.pi / 180, np.pi / 180, 1, 1, 1],
        ).view(1, 6)

        # init loss
        if loss is None:
            loss = {"name": "ncc", "win": None}
        if isinstance(loss, dict):
            loss_name = loss.pop("name")
            params = loss.copy()
            if loss_name == "mse":
                self.loss = types.MethodType(
                    lambda s, x, y: F.mse_loss(x, y, reduction="none", **params), self
                )
            elif loss_name == "ncc":
                self.loss = types.MethodType(
                    lambda s, x, y: ncc_loss(
                        x, y, reduction="none", level=s.current_level, **params
                    ),
                    self,
                )
            else:
                raise Exception("unknown loss")
        elif callable(loss):
            self.loss = types.MethodType(
                lambda s, x, y: cast(Callable, loss)(s, x, y), self
            )
        else:
            raise Exception("unknown loss")

        # init optimizer
        if optimizer is None:
            optimizer = {"name": "gd", "momentum": 0.1}
        if optimizer["name"] == "gd":
            if "momentum" not in optimizer:
                optimizer["momentum"] = 0
        self.optimizer = optimizer

    def degree2rad(self, theta: torch.Tensor) -> torch.Tensor:
        return theta * self._degree2rad

    def rad2degree(self, theta: torch.Tensor) -> torch.Tensor:
        return theta / self._degree2rad

    def clean_optimizer_state(self) -> None:
        if self.optimizer["name"] == "gd":
            if "buf" in self.optimizer:
                self.optimizer.pop("buf")

    def prepare(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> None:
        return

    def forward_tensor(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._degree2rad = self._degree2rad.to(device=theta.device, dtype=theta.dtype)
        self.prepare(theta, source, target, params)
        theta0 = theta.clone()
        theta = self.rad2degree(theta.detach()).requires_grad_(self.auto_grad)
        with torch.set_grad_enabled(self.auto_grad):
            theta, loss = self.multilevel(theta, source, target)
        with torch.no_grad():
            dtheta = self.degree2rad(theta) - theta0
        return theta0 + dtheta, loss

    def update_level(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("")

    def multilevel(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for level in range(self.num_levels - 1, -1, -1):
            self.current_level = level
            source_new, target_new = self.update_level(theta, source, target)
            theta, loss = self.singlelevel(
                theta,
                source_new,
                target_new,
                self.num_steps[level],
                self.step_sizes[level],
            )
            self.clean_optimizer_state()

        return theta, loss

    def singlelevel(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        num_steps: int,
        step_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for _ in range(num_steps):
            theta, loss = self.step(theta, source, target, step_size)
            step_size /= 2
        return theta, loss

    def step(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.activate_idx = torch.ones(
            theta.shape[0], device=theta.device, dtype=torch.bool
        )
        loss_all = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
        for _ in range(self.max_iter):
            theta_a, source_a, target_a = self.activate_set(theta, source, target)
            loss, grad = self.grad(theta_a, source_a, target_a, step_size)
            loss_all[self.activate_idx] = loss

            with torch.no_grad():
                step = self.optimizer_step(grad) * -step_size
                theta_a.add_(step)
                loss_new = self.evaluate(theta_a, source_a, target_a)
                idx_new = loss_new + 1e-4 < loss
                self.activate_idx[self.activate_idx.clone()] = idx_new
                if not torch.any(self.activate_idx):
                    break
                theta[self.activate_idx] += step[idx_new]

        return theta, loss_all.detach()

    def activate_set(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = theta[self.activate_idx]
        if source.shape[0] > 1:
            source = source[self.activate_idx]
        if target.shape[0] > 1:
            target = target[self.activate_idx]
        return theta, source, target

    def grad(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        step_size: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = self.evaluate(theta, source, target)
        if self.auto_grad:
            grad = torch.autograd.grad([loss.sum()], [theta])[0]
        else:
            backup = torch.empty_like(theta[:, 0])
            grad = torch.zeros_like(theta)
            for j in range(theta.shape[1]):
                backup.copy_(theta[:, j])
                theta[:, j].copy_(backup + step_size)
                loss1 = self.evaluate(theta, source, target)
                theta[:, j].copy_(backup - step_size)
                loss2 = self.evaluate(theta, source, target)
                theta[:, j].copy_(backup)
                grad[:, j] = loss1 - loss2
        return loss, grad

    def warp(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("warp")

    def evaluate(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        warpped, target = self.warp(theta, source, target)
        loss = self.loss(warpped, target)
        loss = loss.view(loss.shape[0], -1).mean(1)
        return loss

    def optimizer_step(self, grad: torch.Tensor) -> torch.Tensor:
        if self.optimizer["name"] == "gd":
            step = self.gd_step(grad)
        else:
            raise Exception("unknown optimizer")
        step = step / (torch.linalg.norm(step, dim=-1, keepdim=True) + 1e-6)
        return step

    def gd_step(self, grad: torch.Tensor) -> torch.Tensor:
        if self.optimizer["momentum"]:
            if "buf" not in self.optimizer:
                self.optimizer["buf"] = grad.clone()
            else:
                self.optimizer["buf"][self.activate_idx] = (
                    self.optimizer["buf"][self.activate_idx]
                    * self.optimizer["momentum"]
                    + grad
                )
            return self.optimizer["buf"][self.activate_idx]
        else:
            return grad


class VolumeToVolumeRegistration(Registration):
    """
    体对体配准类：专门用于3D体积之间的刚性配准
    
    继承自Registration基类，实现多分辨率体配准算法（多分辨率防止陷入局部最优）
    使用高斯模糊和重采样实现多尺度配准策略
    """
    trans_first = False  # 变换顺序：先旋转后平移，收敛更快

    
    def update_level( #多分辨率策略
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新当前分辨率层级的图像数据
        
        在多分辨率配准中，每个层级都会对源图像和目标图像进行
        高斯模糊和重采样，以降低分辨率提高配准效率
        
        Args:
            theta: 当前变换参数（旋转平移向量）
            source: 源体积图像 [B, C, D, H, W]
            target: 目标体积图像 [B, C, D, H, W]
            B是批量大小,C是通道数,D/H/W是体素维度
        Returns:
            处理后的源图像和目标图像
        """
        ################### 计算当前层级的高斯模糊标准差
        # 标准差与分辨率成反比，随层级增加而增大
        sigma_source = [
            0.5 * (2**self.current_level) / res for res in self.relative_res_source
        ]
        # 对源图像进行高斯模糊，平滑图像细节
        source = gaussian_blur(source, sigma_source, truncated=4.0)
        

        sigma_target = [
            0.5 * (2**self.current_level) / res for res in self.relative_res_target
        ]
        # 对目标图像进行高斯模糊
        target = gaussian_blur(target, sigma_target, truncated=4.0)

        ################### 重采样到当前分辨率层级
        # 分辨率降低为2^current_level倍
        source = resample(
            source, self.relative_res_source[::-1], [2**self.current_level] * 3
        )
        target = resample(
            target, self.relative_res_target[::-1], [2**self.current_level] * 3
        )

        # 计算新的实际分辨率
        res_new = self.res * (2**self.current_level)

        ################### 创建有效像素掩码（非零像素）
        # 仅对目标图像中存在有效数据的像素进行配准
        mask = (target > 0).view(-1)

        # 生成网格坐标，用于后续的坐标变换
        grid = meshgrid(
            (target.shape[-1], target.shape[-2], target.shape[-3]),  # W, H, D
            (res_new, res_new, res_new),  # 各向同性分辨率
            device=target.device,
        )
        # 展平网格并应用掩码，只保留有效像素
        grid = grid.reshape(-1, 3)[mask, :]
        self._grid = grid  # 保存网格用于warp操作

        # 展平目标图像并应用掩码
        self._target_flat = target.view(-1)[mask]

        # 计算网格缩放因子，用于将物理坐标映射到[-1,1]范围
        scale = torch.tensor(
            [
                2.0 / (source.shape[-1] - 1),  # X方向缩放
                2.0 / (source.shape[-2] - 1),  # Y方向缩放
                2.0 / (source.shape[-3] - 1),  # Z方向缩放
            ],
            device=source.device,
            dtype=source.dtype,
        )
        # 根据分辨率调整缩放因子
        self._grid_scale = scale / res_new

        return source, target

    def warp(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对源图像进行坐标变换（扭曲）
        
        使用刚性变换矩阵将源图像变换到目标图像的空间
        
        Args:
            theta: 变换参数（6维向量：3旋转+3平移）
            source: 源体积图像
            target: 目标体积图像（用于形状参考）
            
        Returns:
            变换后的源图像和对应的目标图像区域
        """
        # 将角度参数转换为刚性变换矩阵
        mat = (
            RigidTransform(self.degree2rad(theta), trans_first=self.trans_first)
            .inv()  # 计算逆变换：从目标到源
            .matrix()  # 获取4x4变换矩阵
        )
        
        # 应用变换矩阵到网格坐标
        # 将3D坐标转换为齐次坐标并应用变换
        grid = torch.matmul(
            mat[:, :, :-1],  # 旋转部分（3x3）
            self._grid.reshape(-1, 3, 1) + mat[:, :, -1:]  # 坐标+平移
        )
        
        # 重新整形为grid_sample需要的格式 [B, D, H, W, 3]
        grid = grid.reshape(1, -1, 1, 1, 3)
        # 使用双线性插值进行图像变换
        warpped = F.grid_sample(source, grid * self._grid_scale, align_corners=True)
        
        # 返回展平的变换后图像和目标图像
        return warpped.view(1, 1, -1), self._target_flat.view(1, 1, -1)

    def prepare(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> None:
        """
        准备配准参数
        
        计算源图像和目标图像的相对分辨率，用于多尺度处理
        
        Args:
            theta: 初始变换参数
            source: 源图像
            target: 目标图像
            params: 包含分辨率信息的参数字典
        """
        # 验证输入维度（应为5维：[B, C, D, H, W]）
        assert source.ndim == 5 and target.ndim == 5
        
        # 获取源和目标的分辨率（Z, Y, X方向）
        res_source = params["res_source"]
        res_target = params["res_target"]
        
        # 计算最小分辨率作为基准
        self.res = min(res_source + res_target)
        # 计算相对分辨率（相对于最小分辨率）
        self.relative_res_source = [r / self.res for r in res_source]
        self.relative_res_target = [r / self.res for r in res_target]

    def forward(
        self,
        source: Union[Stack, Volume],
        target: Union[Stack, Volume],
        use_mask: bool = False,
    ) -> Tuple[RigidTransform, torch.Tensor]:
        """
        前向传播：执行完整的体对体配准流程
        
        Args:
            source: 源堆栈或体积
            target: 目标堆栈或体积
            use_mask: 是否使用掩码进行配准
            
        Returns:
            transform_out: 配准后的变换矩阵
            loss: 最终的配准损失值
        """
        # 如果输入是Stack类型，转换为Volume类型
        if isinstance(source, Stack):
            source = source.get_volume(copy=False)
        if isinstance(target, Stack):
            target = target.get_volume(copy=False)
            
        # 构建分辨率参数字典
        params = {
            "res_source": [
                source.resolution_z,  # Z方向分辨率
                source.resolution_y,  # Y方向分辨率
                source.resolution_x,  # X方向分辨率
            ],
            "res_target": [
                target.resolution_z,
                target.resolution_y,
                target.resolution_x,
            ],
        }
        
        # 计算初始变换参数：从源到目标的相对变换
        theta = (
            target.transformation.inv()  # 目标逆变换
            .compose(source.transformation)  # 组合源变换
            .axisangle(self.trans_first)  # 转换为轴角表示
        )

        # 根据是否使用掩码准备输入数据
        if use_mask:
            source_input = source.image * source.mask  # 应用掩码
            target_input = target.image * target.mask
        else:
            source_input = source.image  # 原始图像
            target_input = target.image

        # 调用父类的forward_tensor方法进行多分辨率配准
        theta, loss = self.forward_tensor(
            theta, source_input[None, None], target_input[None, None], params
        )

        # 构建最终的输出变换矩阵
        transform_out = target.transformation.compose(
            RigidTransform(theta, trans_first=self.trans_first)
        )

        return transform_out, loss


class SliceToVolumeRegistration(Registration):
    """
    切片到体积配准类：专门用于将2D切片配准到3D体积的配准算法
    
    继承自Registration基类，实现了多分辨率配准框架，专门处理切片采集和体积重建场景。
    主要用于医学图像处理中的切片到体积配准问题。
    """
    trans_first = True  # 变换顺序标志：由于slice_acquisition模块的设计，使用平移优先的变换顺序

    def update_level(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新多分辨率层级：在当前层级对源和目标图像进行预处理
        
        Args:
            theta: 当前变换参数（6自由度刚性变换参数）
            source: 源图像（体积数据）
            target: 目标图像（切片数据）
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 处理后的源图像和目标图像
        """
        # 计算当前层级的高斯模糊标准差
        sigma = 0.5 * (2**self.current_level)
        # 对源和目标图像进行高斯模糊，减少高频噪声
        source = gaussian_blur(source, sigma, truncated=4.0)
        target = gaussian_blur(target, sigma, truncated=4.0)
        # 对目标图像进行下采样，实现多分辨率策略
        target = resample(target, [1] * 2, [2**self.current_level] * 2)
        # 如果存在切片掩码，对其进行相应的重采样处理
        self.slices_mask_resampled = (
            (
                resample(
                    self.slices_mask.float(), [1] * 2, [2**self.current_level] * 2
                )
                > 0
            )
            if self.slices_mask is not None
            else None
        )
        return source, target

    def prepare(
        self,
        theta: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        params: Dict[str, Any],
    ) -> None:
        """
        配准准备：初始化配准所需的参数和配置
        
        Args:
            theta: 变换参数
            source: 源图像（体积）
            target: 目标图像（切片）
            params: 参数字典，包含分辨率等信息
        """
        # 初始化点扩散函数（PSF），这里使用单位矩阵，意味着不考虑点扩散效应
        self.psf = torch.ones((1, 1, 1), device=theta.device, dtype=theta.dtype)
        # 从参数中获取切片分辨率和体积分辨率
        self.res_s = params["res_s"]  # 切片分辨率
        self.res_v = params["res_r"]  # 体积分辨率
        # 验证输入张量的维度：源应为5D（batch, channel, depth, height, width），体积
        assert len(source.shape) == 5
        # 目标应为4D（batch, channel, height, width），切片
        assert len(target.shape) == 4

    def warp(
        self, theta: torch.Tensor, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        图像变换：应用刚性变换将源体积投影到切片空间
        
        Args:
            theta: 变换参数（6自由度）
            source: 源体积图像
            target: 目标切片图像
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 变换后的图像和原始目标图像
        """
        # 将角度参数转换为旋转矩阵
        transforms = axisangle2mat(self.degree2rad(theta))
        # 更新变换矩阵的分辨率，使其适应体积分辨率
        transforms = mat_update_resolution(transforms, 1, self.res_v)
        volume = source
        slices = target
        slices_mask: Optional[torch.Tensor]
        # 如果存在重采样后的切片掩码，应用掩码到切片
        if self.slices_mask_resampled is not None:
            slices_mask = self.slices_mask_resampled[self.activate_idx]
            slices = slices * slices_mask
        else:
            slices_mask = None
        # 使用切片采集模块将体积投影到切片空间
        warpped = slice_acquisition(
            transforms,
            volume,
            self.volume_mask,
            slices_mask,
            self.psf,
            slices.shape[-2:],
            self.res_s * (2**self.current_level) / self.res_v,
            False,
            False,
        )
        return warpped, slices # 变换后的图像和原始目标图像

    def forward(
        self,
        stack: Stack,
        volume: Volume,
        use_mask: bool = False,
    ) -> Tuple[RigidTransform, torch.Tensor]:
        """
        前向传播：执行切片到体积的配准过程
        
        Args:
            stack: 输入切片堆栈对象
            volume: 输入体积对象
            use_mask: 是否使用掩码进行配准
            
        Returns:
            Tuple[RigidTransform, torch.Tensor]: 配准后的变换矩阵和损失值
        """
        eps = 1e-3
        # 验证输入体积是否为各向同性（各方向分辨率相等）
        assert (
            abs(volume.resolution_x - volume.resolution_y) < eps
            and abs(volume.resolution_x - volume.resolution_z) < eps
        ), "input volume should be isotropic!"
        # 验证输入切片是否为各向同性
        assert (
            abs(stack.resolution_x - stack.resolution_y) < eps
        ), "input slices should be isotropic!"

        # 设置配准参数：切片分辨率和体积分辨率
        params = {"res_s": stack.resolution_x, "res_r": volume.resolution_x}

        # 获取切片和体积的初始变换矩阵
        slices_transform = stack.transformation
        volume_transform = volume.transformation

        # 计算相对变换：将切片变换转换到体积坐标系下
        slices_transform = volume_transform.inv().compose(slices_transform)
        # 将变换矩阵转换为轴角表示
        theta = slices_transform.axisangle(self.trans_first)

        # 设置掩码（如果启用）
        self.volume_mask = volume.mask[None, None] if use_mask else None
        self.slices_mask = stack.mask if use_mask else None

        # 执行张量级别的配准前向传播
        theta, loss = self.forward_tensor(
            theta, volume.image[None, None], stack.slices, params
        )

        # 构建最终的输出变换矩阵
        transform_out = RigidTransform(theta, trans_first=self.trans_first)
        # 将相对变换组合回原始体积坐标系
        transform_out = volume_transform.compose(transform_out)

        return transform_out, loss


def stack_registration(
    source_stacks: List[List[Stack]],
    centering: bool = False,
    args_registration: Optional[Dict] = None,
) -> List[Stack]:
    """
    堆栈配准函数：将多个堆栈列表对齐到目标堆栈
    
    Args:
        source_stacks: 源堆栈列表，每个元素是一个堆栈列表，表示同一层的不同视角
        centering: 是否进行中心化处理，默认为False
        args_registration: 可选的配准参数字典，用于覆盖默认参数
    
    Returns:
        List[Stack]: 配准后的堆栈列表
    """
    # 堆栈配准
    # 设置默认的体对体配准参数
    vvr_args: Dict[str, Any] = {
        "num_levels": 3,      # 多分辨率层数
        "num_steps": 4,       # 每层迭代步数
        "step_size": 2,       # 步长
        "max_iter": 20,       # 最大迭代次数
    }
    # 如果提供了自定义参数，更新默认参数
    if args_registration is not None:
        vvr_args.update(args_registration)
    # 创建体对体配准器实例
    vvr = VolumeToVolumeRegistration(**vvr_args)

    # 选择第一个堆栈列表的第一个堆栈作为目标
    target_stack = source_stacks[0][0]
    # 获取目标的体积表示
    target = target_stack.get_volume(copy=False)
    # 将所有源堆栈转换为体积表示
    sources = [[s.get_volume(copy=False) for s in ss] for ss in source_stacks]

    # 获取堆栈列表数量和每个列表中的堆栈数量
    n_lists = len(sources)
    n_stacks = len(sources[0])

    # 存储配准后的变换矩阵和堆栈
    ts_registered: List[RigidTransform] = []
    stacks_out: List[Stack] = []

    # 遍历每个堆栈位置（同一视角的不同层）
    for j in range(n_stacks):
        if j == 0:
            # 第一个堆栈作为参考，直接使用其变换
            ts_registered.append(target.transformation)
            stacks_out.append(target_stack)
        else:
            # 对于其他堆栈，选择配准效果最好的
            ncc_min: Union[float, torch.Tensor] = float("inf")  # 初始化最小NCC值为无穷大
            # 遍历所有堆栈列表，找到配准效果最好的堆栈
            for k in range(n_lists):
                # 将当前堆栈的变换相对于参考堆栈进行对齐
                sources[k][j].transformation = (
                    ts_registered[0]
                    .compose(sources[k][0].transformation.inv())
                    .compose(sources[k][j].transformation)
                ) # compose: 先对齐到参考，再对齐到当前层, 当计算出新的 theta（6参数：3旋转+3平移）后，通过 compose 方法更新切片位置 
                # 执行配准，获取变换矩阵和NCC值
                t_cur, ncc = vvr(sources[k][j], target, use_mask=True)
                # 如果当前NCC值更小（配准效果更好），更新最佳结果
                if ncc < ncc_min:
                    ncc_min, t_best, s_best = ncc, t_cur, source_stacks[k][j]
            # 记录最佳配准结果
            ts_registered.append(t_best)
            stacks_out.append(s_best)

    # 如果需要进行中心化处理
    if centering:
        # 计算中心化变换：将第一个变换的平移分量取反
        t_center_ax = ts_registered[0].axisangle(trans_first=False).clone()
        t_center_ax[..., :3] = 0  # 旋转分量为0
        t_center_ax[..., 3:] *= -1  # 平移分量取反
        t_center = RigidTransform(t_center_ax)

    # 应用最终的变换到所有堆栈
    for s, t in zip(stacks_out, ts_registered):
        # 获取堆栈的初始变换
        transform_init = s.init_stack_transform()
        # 组合配准变换和初始变换
        transform_out = t.compose(transform_init)
        # 如果启用了中心化，应用中心化变换
        if centering:
            transform_out = t_center.compose(transform_out)
        # 更新堆栈的变换矩阵
        s.transformation = transform_out

    return stacks_out