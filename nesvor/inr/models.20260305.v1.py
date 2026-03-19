from argparse import Namespace
from math import log2
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, Tuple
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from .hash_grid_torch import HashEmbedder
from ..transform import RigidTransform, ax_transform_points, mat_transform_points
from ..utils import resolution2sigma
from .kplanes_encoder import KPlanesFeatureEncoder

USE_TORCH = False

if not USE_TORCH:
    try:
        import tinycudann as tcnn
    except:
        logging.warning("Fail to load tinycudann. Will use pytorch implementation.")
        USE_TORCH = True


# key for loss/regularization
D_LOSS = "MSE"
S_LOSS = "logVar"
DS_LOSS = "MSE+logVar"
B_REG = "biasReg"
T_REG = "transReg"
I_REG = "imageReg"
D_REG = "deformReg"


def build_encoding(**config):
    if USE_TORCH:
        encoding = HashEmbedder(**config)
    else:
        n_input_dims = config.pop("n_input_dims")
        dtype = config.pop("dtype")
        try:
            encoding = tcnn.Encoding(
                n_input_dims=n_input_dims, encoding_config=config, dtype=dtype
            )
        except RuntimeError as e:
            if "TCNN was not compiled with half-precision support" in str(e):
                logging.error(
                    "TCNN was not compiled with half-precision support! "
                    "Try using --single-precision in the nesvor command! "
                )
            raise e
    return encoding


def build_network(**config):
    """
    构建神经网络模型的通用工厂函数
    
    根据配置参数构建MLP网络，支持两种实现方式：
    1. 使用tiny-cuda-nn的高性能实现（半精度）
    2. 使用PyTorch的标准实现（单精度/半精度）
    
    Args:
        **config: 网络配置参数字典，包含：
            - dtype: 数据类型（torch.float16或torch.float32）
            - n_input_dims: 输入维度
            - n_output_dims: 输出维度
            - activation: 隐藏层激活函数名称
            - output_activation: 输出层激活函数名称
            - n_neurons: 隐藏层神经元数量
            - n_hidden_layers: 隐藏层数量
            
    Returns:
        nn.Module: 构建好的神经网络模型
    """
    dtype = config.pop("dtype")  # 提取并移除数据类型参数
    assert dtype == torch.float16 or dtype == torch.float32  # 验证数据类型
    
    # 使用tiny-cuda-nn的高性能实现（仅支持半精度且CUDA可用时）
    if dtype == torch.float16 and not USE_TORCH:
        return tcnn.Network(
            n_input_dims=config["n_input_dims"],
            n_output_dims=config["n_output_dims"],
            network_config={
                "otype": "CutlassMLP",  # 使用Cutlass优化的MLP
                "activation": config["activation"],
                "output_activation": config["output_activation"],
                "n_neurons": config["n_neurons"],
                "n_hidden_layers": config["n_hidden_layers"],
            },
        )
    else:
        # 使用PyTorch标准实现
        
        # 解析激活函数：将字符串转换为实际的激活函数类
        activation = (
            None
            if config["activation"] == "None"  # "None"表示无激活函数
            else getattr(nn, config["activation"])  # 通过反射获取激活函数类
        )
        output_activation = (
            None
            if config["output_activation"] == "None"
            else getattr(nn, config["output_activation"])
        )
        
        models = []  # 存储网络层的列表
        
        # 构建隐藏层
        if config["n_hidden_layers"] > 0:
            # 输入层：从输入维度到隐藏层维度
            models.append(nn.Linear(config["n_input_dims"], config["n_neurons"]))
            
            # 中间隐藏层：保持相同维度
            for _ in range(config["n_hidden_layers"] - 1):
                if activation is not None:
                    models.append(activation())  # 添加激活函数
                models.append(nn.Linear(config["n_neurons"], config["n_neurons"]))  # 线性层
            
            # 最后一层隐藏层的激活函数
            if activation is not None:
                models.append(activation())
            
            # 输出层：从隐藏层维度到输出维度
            models.append(nn.Linear(config["n_neurons"], config["n_output_dims"]))
        else:
            # 无隐藏层：直接连接输入输出
            models.append(nn.Linear(config["n_input_dims"], config["n_output_dims"]))
        
        # 输出层激活函数
        if output_activation is not None:
            models.append(output_activation())
        
        # 将所有层组合成顺序模型
        return nn.Sequential(*models)


def compute_resolution_nlevel(
    bounding_box: torch.Tensor,
    coarsest_resolution: float,
    finest_resolution: float,
    level_scale: float,
    spatial_scaling: float,
) -> Tuple[int, int]:
    base_resolution = (
        (
            (bounding_box[1] - bounding_box[0]).max()
            * spatial_scaling
            / coarsest_resolution
        )
        .ceil()
        .int()
        .item()
    )
    n_levels = (
        (
            torch.log2(
                (bounding_box[1] - bounding_box[0]).max()
                * spatial_scaling
                / finest_resolution
                / base_resolution
            )
            / log2(level_scale)
            + 1
        )
        .ceil()
        .int()
        .item()
    )
    return int(base_resolution), int(n_levels)

def compute_plane_tv(t):
    """K-Planes 原版一阶 TV 计算"""
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)

def compute_plane_smoothness(t):
    """K-Planes 原版二阶导数时间平滑度"""
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2 (h)
    first_difference = t[..., 1:, :] - t[..., :h-1, :]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]
    return torch.square(second_difference).mean()


class INR(nn.Module):
    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        # 注意：外部传入的 bounding_box 必须是 4D 的！形如 [[min_x...t], [max_x...t]]
        self.register_buffer("bounding_box", bounding_box)
        
        # 1. 接入 4D K-Planes 物理分流编码器
        self.encoder = KPlanesFeatureEncoder(
            aabb=self.bounding_box,
            # 这里硬编码了网格的基础配置，你可以根据显存和需求调整 resolution
            grid_config=[{"grid_dimensions": 2, "input_coordinate_dim": 4, 
                          "output_coordinate_dim": args.n_features_per_level, 
                          "resolution": [64, 64, 64, 25]}], 
            concat_features_across_scales=True,
            multiscale_res=[1, 2, 4, 8], # 多尺度倍率
            v3d_scale_indices=[0, 1]     # 【魔法阵眼】：V_3D 仅使用前两个低频尺度！
        )

        # 2. Density Net / 解剖网络 (吃全频段的 V_4D)
        self.density_net = build_network(
            n_input_dims=self.encoder.v4d_feature_dim,
            n_output_dims=1 + args.n_features_z, # 1 是密度，剩下的留给 Sigma Net
            activation="ReLU",
            output_activation="None",
            n_neurons=args.width,
            n_hidden_layers=args.depth,
            dtype=torch.float32 if getattr(args, "img_reg_autodiff", False) else args.dtype,
        )
        
        logging.debug(
            f"K-Planes initialized. V_4D dim={self.encoder.v4d_feature_dim}, V_3D dim={self.encoder.v3d_feature_dim}"
        )

    def forward(self, xyzt: torch.Tensor):
        # 1. 通过纯编码器获取分流特征
        v_3d, v_4d = self.encoder(xyzt)
        
        if not self.training:
            v_3d = v_3d.to(dtype=xyzt.dtype)
            v_4d = v_4d.to(dtype=xyzt.dtype)
            
        # 2. 解剖网络解码 V_4D
        z = self.density_net(v_4d)
        
        # 提取第一个通道作为密度值，并做 Softplus 保证非负
        density = F.softplus(z[..., 0]) 
        
        if self.training:
            # 将 v_3d 和 v_4d 一起吐出，交给下游网络（b_net 和 sigma_net）
            return density, v_3d, v_4d, z
        else:
            return density, v_3d, v_4d, z

class DeformNet(nn.Module):
    """
    变形网络(Deformation Network)
    
    用于学习3D空间中的非刚性变形场，将变形后的坐标映射回原始坐标空间。
    在NeSVoR模型中，该网络用于补偿切片间的非刚性变形，提高重建精度。
    
    核心思想：
    - 输入：3D坐标 + 切片嵌入特征（变形）
    - 输出：变形后的3D坐标
    - 使用残差连接：变形量 = 网络输出 + 原始坐标
    """
    
    def __init__(
        self, bounding_box: torch.Tensor, args: Namespace, spatial_scaling: float = 1.0
    ) -> None:
        """
        初始化变形网络
        
        Args:
            bounding_box: 重建区域的边界框 [2, 3]，定义3D空间范围
            args: 命令行参数命名空间，包含网络配置参数
            spatial_scaling: 空间缩放因子，用于调整分辨率计算
        """
        super().__init__()
        if TYPE_CHECKING:
            self.bounding_box: torch.Tensor
        # 注册边界框为缓冲区（不参与梯度计算）
        self.register_buffer("bounding_box", bounding_box)

        # 计算哈希网格编码的分辨率和层级数
        base_resolution, n_levels = compute_resolution_nlevel(
            bounding_box,
            args.coarsest_resolution_deform,  # 变形网络的最粗分辨率
            args.finest_resolution_deform,    # 变形网络的最细分辨率
            args.level_scale_deform,          # 变形网络的层级间缩放比例
            spatial_scaling,                   # 空间缩放因子
        )
        level_scale = args.level_scale_deform

        # 构建哈希网格编码器：将3D坐标映射到高维特征空间
        self.encoding = build_encoding(
            n_input_dims=3,  # 输入维度：3D坐标(x,y,z)
            otype="HashGrid",  # 使用哈希网格编码
            n_levels=n_levels,  # 多分辨率层级数
            n_features_per_level=args.n_features_per_level_deform,  # 每层特征维度
            log2_hashmap_size=args.log2_hashmap_size,  # 哈希表大小（对数）
            base_resolution=base_resolution,  # 基础网格分辨率
            per_level_scale=level_scale,  # 层级间缩放比例
            dtype=args.dtype,  # 数据类型（半精度/单精度）
            interpolation="Smoothstep",  # 平滑插值，提供更连续的变形场
        )
        
        # 构建变形网络：MLP网络预测3D变形向量
        self.deform_net = build_network(
            n_input_dims=n_levels * args.n_features_per_level_deform
            + args.n_features_deform,  # 输入：位置编码特征 + 切片嵌入特征（变形）
            n_output_dims=3,  # 输出：3D变形向量(dx, dy, dz)
            activation="Tanh",  # Tanh激活函数，限制变形量在[-1,1]范围内
            output_activation="None",  # 输出层无激活（原始输出）
            n_neurons=args.width,  # 隐藏层宽度
            n_hidden_layers=2,  # 固定2层隐藏层，保持网络轻量
            dtype=torch.float32,  # 使用单精度，确保数值稳定性
        )
        
        # 初始化网络权重：使用小范围均匀分布，避免初始变形过大
        for p in self.deform_net.parameters():
            torch.nn.init.uniform_(p, a=-1e-4, b=1e-4)
        
        # 记录哈希网格编码的超参数信息
        logging.debug(
            "hyperparameters for hash grid encoding (deform net): "
            + "lowest_grid_size=%d, highest_grid_size=%d, scale=%1.2f, n_levels=%d",
            base_resolution,  # 最粗网格大小
            int(base_resolution * level_scale ** (n_levels - 1)),  # 最细网格大小
            level_scale,  # 缩放比例
            n_levels,  # 层级数
        )

    def forward(self, x: torch.Tensor, e: torch.Tensor):
        """
        变形网络的前向传播过程
        
        将输入的3D坐标通过变形网络映射到变形后的坐标空间。
        使用残差连接：变形坐标 = 原始坐标 + 网络预测的变形量
        
        Args:
            x: 输入坐标张量 [..., 3]，可以是任意形状的3D坐标
            e: 切片嵌入特征张量 [..., n_features_deform]，提供切片特定的变形信息
            
        Returns:
            torch.Tensor: 变形后的坐标张量 [..., 3]，保持与输入相同的形状
        """
        # 坐标归一化：将世界坐标系下的坐标映射到[0,1]范围
        x = (x - self.bounding_box[0]) / (self.bounding_box[1] - self.bounding_box[0])
        
        # 保存原始形状（排除最后一个维度）
        x_shape = x.shape
        
        # 展平坐标张量：将任意形状转换为二维 [batch*..., 3]
        x = x.view(-1, x.shape[-1])
        
        # 计算位置编码：通过哈希网格将3D坐标映射到高维特征空间
        pe = self.encoding(x)
        
        # 拼接输入特征：位置编码 + 切片嵌入特征
        inputs = torch.cat((pe, e.reshape(-1, e.shape[-1])), -1)
        
        # 变形网络前向传播：预测变形量，并使用残差连接
        # 变形量 = 网络输出 + 原始坐标（残差连接确保变形平滑）
        outputs = self.deform_net(inputs) + x
        
        # 坐标反归一化：将[0,1]范围的坐标映射回世界坐标系
        outputs = (
            outputs * (self.bounding_box[1] - self.bounding_box[0])
            + self.bounding_box[0]
        )
        
        # 恢复原始形状
        return outputs.view(x_shape)


class NeSVoR(nn.Module):
    """
    NeSVoR (Neural Slice-to-Volume Reconstruction) 主模型类
    
    这是一个基于隐式神经表示(INR)的医学图像三维重建模型，
    用于从二维切片序列重建三维体数据。
    """
    
    def __init__(
        self,
        transformation: RigidTransform,
        resolution: torch.Tensor,
        v_mean: float,
        bounding_box: torch.Tensor,
        spatial_scaling: float,
        args: Namespace,
    ) -> None:
        """
        初始化NeSVoR模型
        
        Args:
            transformation: 刚性变换参数，用于切片配准
            resolution: 图像分辨率张量
            v_mean: 图像强度均值，用于正则化参数计算
            bounding_box: 重建区域的边界框
            spatial_scaling: 空间缩放因子
            args: 命令行参数命名空间
        """
        super().__init__()
        if "cpu" in str(args.device):  # CPU模式
            global USE_TORCH
            USE_TORCH = True
        self.spatial_scaling = spatial_scaling
        self.args = args
        self.n_slices = 0  # 切片数量，将在transformation setter中初始化
        self.trans_first = True  # 变换顺序标志
        self.transformation = transformation  # 设置变换参数
        self.psf_sigma = resolution2sigma(resolution, isotropic=False)  # 计算点扩散函数标准差
        self.delta = args.delta * v_mean  # 边缘感知正则化参数
        self.build_network(bounding_box)  # 构建网络组件
        self.to(args.device)  # 移动到指定设备

    @property
    def transformation(self) -> RigidTransform: #定义transformation属性的getter方法,即读取逻辑，如current_transform = model.transformation  # 调用getter
        """
        获取当前变换参数
        
        Returns:
            RigidTransform: 当前的刚性变换对象
        """
        return RigidTransform(self.axisangle.detach(), self.trans_first)

    @transformation.setter
    def transformation(self, value: RigidTransform) -> None: #定义transformation属性的setter方法,即写入逻辑，如model.transformation = new_transform  # 调用setter
        """
        设置变换参数
        
        Args:
            value: 新的刚性变换对象
        """
        if self.n_slices == 0:
            self.n_slices = len(value)
        else:
            assert self.n_slices == len(value)
        axisangle = value.axisangle(self.trans_first)
        if TYPE_CHECKING:
            self.axisangle_init: torch.Tensor
        self.register_buffer("axisangle_init", axisangle.detach().clone())
        if not self.args.no_transformation_optimization:
            self.axisangle = nn.Parameter(axisangle.detach().clone()) #将axisangle转换为可训练参数
        else:
            self.register_buffer("axisangle", axisangle.detach().clone())

    def build_network(self, bounding_box) -> None:
        """
        构建网络的所有组件
        
        Args:
            bounding_box: 重建区域的边界框
        """
        # 切片嵌入层：为每个切片学习特征表示
        if self.args.n_features_slice:
            self.slice_embedding = nn.Embedding(
                self.n_slices, self.args.n_features_slice
            )
        
        # 切片尺度参数：学习每个切片的强度缩放因子
        if not self.args.no_slice_scale:
            self.logit_coef = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        
        # 切片方差参数：学习每个切片的噪声方差
        if not self.args.no_slice_variance:
            self.log_var_slice = nn.Parameter(
                torch.zeros(self.n_slices, dtype=torch.float32)
            )
        
        # 【强制关闭 DeformNet】：在 4D 架构第一阶段，封印该模块防止形变歧义！
        if self.args.deformable:
            logging.warning("Deformable module is explicitly IGNORED in 4D K-Planes Phase 1!")
            self.args.deformable = False
        
        # 隐式神经表示(INR)网络：学习3D密度场
        self.inr = INR(bounding_box, self.args, self.spatial_scaling)
        
        # 修改：Sigma net (方差网络，吃 V_4D 和前置特征)
        if not self.args.no_pixel_variance:
            self.sigma_net = build_network(
                n_input_dims=self.args.n_features_slice + self.args.n_features_z + self.inr.encoder.v4d_feature_dim,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )
            
        # 修改：Bias net (偏置场网络，绝对绝缘时间，仅吃 V_3D)
        if self.args.n_levels_bias:
            self.b_net = build_network(
                n_input_dims=self.inr.encoder.v3d_feature_dim + self.args.n_features_slice,
                n_output_dims=1,
                activation="ReLU",
                output_activation="None",
                n_neurons=self.args.width,
                n_hidden_layers=self.args.depth,
                dtype=self.args.dtype,
            )

    def forward(
            self,
            xyzt: torch.Tensor,       # [B, 4] DataLoader 传来的时空坐标
            v: torch.Tensor,          # [B, 1] 真实像素值
            slice_idx: torch.Tensor,  # [B, 1] 切片索引
            slice_R: torch.Tensor,    # [B, 3, 3] 局部->物理世界的旋转矩阵
        ) -> Dict[str, Any]:
            batch_size = xyzt.shape[0]
            n_samples = self.args.n_samples

            # 【修复 2】：DataLoader 传来的 slice_idx 是 [B, 1]，必须压缩成 1D 的 [B] 才能正确查表！
            slice_idx = slice_idx.squeeze(-1)
            
            # 1. 严格剥离时间维度
            xyz = xyzt[:, :3] # [B, 3]
            t = xyzt[:, 3:]   # [B, 1]

            # 2. 局部各向异性 PSF 采样
            xyz_psf_local = torch.randn(batch_size, n_samples, 3, dtype=xyz.dtype, device=xyz.device)
            psf_sigma = self.psf_sigma.to(xyz.device)[slice_idx] # [B, 3] (对应层厚、面内分辨率)
            xyz_psf_local = xyz_psf_local * psf_sigma.unsqueeze(1) # [B, n_samples, 3]
            
            # 将局部扰动通过 slice_R 旋转对齐到当前切片的真实物理法线上
            xyz_psf_world = torch.einsum('bij,bsj->bsi', slice_R, xyz_psf_local)

            # 3. 刚性位姿优化 (Pose T 变换，仅作用于空间坐标)
            t_pose = self.axisangle[slice_idx][:, None] # [B, 1, 6]
            # 注意：ax_transform_points 内部会将 xyz_psf_world 加到 xyz_center 上
            xyz_transformed = ax_transform_points(
                t_pose, xyz[:, None] + xyz_psf_world, self.trans_first
            ) # [B, n_samples, 3]

            # 4. 时空重组：将绝对安全的时间 t 拼回去
            t_expanded = t.unsqueeze(1).expand(batch_size, n_samples, 1)
            xyzt_sampled = torch.cat([xyz_transformed, t_expanded], dim=-1) # [B, n_samples, 4]

            # 5. 提取切片独立特征 (Slice Embedding)
            if self.args.n_features_slice:
                se = self.slice_embedding(slice_idx)[:, None].expand(-1, n_samples, -1)
            else:
                se = None
                
            # 6. 送入我们刚刚改好的网络主通道
            results = self.net_forward(xyzt_sampled, se)
            
            # 提取预测结果
            density = results["density"]
            if "log_bias" in results:
                bias = results["log_bias"].exp()
                bias_detach = bias.detach()
            else:
                log_bias = 0
                bias = 1
                bias_detach = 1
                
            if "log_var" in results:
                var = results["log_var"].exp()
            else:
                var = 1

            # 成像积分 (沿着 n_samples 维度求平均)
            if not self.args.no_slice_scale:
                c: Any = F.softmax(self.logit_coef, 0)[slice_idx] * self.n_slices
            else:
                c = 1
                
            v_out = (bias * density).mean(-1)
            v_out = c * v_out
            
            # 运动方差积分
            if not self.args.no_pixel_variance:
                var = (bias_detach * var).mean(-1)
                var = c.detach() * var
                var = var**2
            if not self.args.no_slice_variance:
                var = var + self.log_var_slice.exp()[slice_idx]

            # 7. 计算基础 Loss 体系 (保真度与位姿)
            losses = {D_LOSS: ((v_out - v) ** 2 / (2 * var)).mean()}
            if not (self.args.no_pixel_variance and self.args.no_slice_variance):
                losses[S_LOSS] = 0.5 * var.log().mean()
                losses[DS_LOSS] = losses[D_LOSS] + losses[S_LOSS]
                
            if not self.args.no_transformation_optimization:
                losses[T_REG] = self.trans_loss(trans_first=self.trans_first)
                
            if self.args.n_levels_bias:
                losses[B_REG] = results.get("log_bias", torch.tensor(0.0)).mean() ** 2
                
            # ==========================================
            # 8. 引入 K-Planes 原生时空正则化
            # ==========================================
            tv_plane, smooth_time, l1_time = self.kplanes_reg()
            
            # 参考 K-Planes 论文针对动态场景 (DyNeRF) 的默认权重建议
            weight_tv = getattr(self.args, 'weight_tv', 0.0001)         # PlaneTV 权重
            weight_smooth = getattr(self.args, 'weight_smooth', 0.001)  # TimeSmoothness 权重 (通常较大)
            weight_l1_time = getattr(self.args, 'weight_l1_time', 0.0001) # L1TimePlanes 权重
            
            losses["TV_Plane"] = tv_plane * weight_tv
            losses["Smooth_Time"] = smooth_time * weight_smooth
            losses["L1_Time"] = l1_time * weight_l1_time
            
            # 伪装并入总 Loss
            losses[I_REG] = losses["TV_Plane"] + losses["Smooth_Time"] + losses["L1_Time"]

            return losses
    def net_forward(
        self,
        xyzt: torch.Tensor,
        se: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        网络前向传播的核心逻辑
        
        Args:
            xyzt: 输入坐标 [..., 4]
            se: 切片嵌入特征 [..., n_features_slice]
            
        Returns:
            Dict[str, Any]: 包含密度、偏置、方差等输出的字典
        """
        # 从 INR 中接住分流特征
        density, v_3d, v_4d, z = self.inr(xyzt)
        prefix_shape = density.shape
        results = {"density": density}

        zs = []
        if se is not None:
            # 把 batch 和 n_samples 展平对接
            zs.append(se.reshape(-1, se.shape[-1]))

        # Bias Field (偏置场)：喂入切片特征 + 静态空间流 V_3D
        if self.args.n_levels_bias:
            results["log_bias"] = self.b_net(torch.cat(zs + [v_3d], -1)).view(prefix_shape)

        # Sigma Net (运动伪影方差)：喂入切片特征 + 解剖隐含特征 z + 动态时空流 V_4D
        if not self.args.no_pixel_variance:
            zs.append(z[..., 1:])
            results["log_var"] = self.sigma_net(torch.cat(zs + [v_4d], -1)).view(prefix_shape)

        return results

    def trans_loss(self, trans_first: bool = True) -> torch.Tensor:
        """
        计算变换参数的正则化损失
        
        Args:
            trans_first: 变换顺序标志
            
        Returns:
            torch.Tensor: 变换正则化损失值
        """
        x = RigidTransform(self.axisangle, trans_first=trans_first)  # 当前变换
        y = RigidTransform(self.axisangle_init, trans_first=trans_first)  # 初始变换
        err = y.inv().compose(x).axisangle(trans_first=trans_first)  # 计算相对变换误差
        loss_R = torch.mean(err[:, :3] ** 2)  # 旋转误差（前3个参数）
        loss_T = torch.mean(err[:, 3:] ** 2)  # 平移误差（后3个参数）
        return loss_R + 1e-3 * self.spatial_scaling * self.spatial_scaling * loss_T  # 加权总损失

    def kplanes_reg(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            完全还原 K-Planes 源码的时空正则化
            """
            tv_plane = torch.tensor(0.0, device=self.axisangle.device)
            smooth_time = torch.tensor(0.0, device=self.axisangle.device)
            l1_time = torch.tensor(0.0, device=self.axisangle.device)
            
            # itertools.combinations(range(4), 2) 对应的索引映射：
            # 0:xy, 1:xz, 2:xt, 3:yz, 4:yt, 5:zt
            spatial_grids = [0, 1, 3] # 原码中的纯空间平面
            time_grids = [2, 4, 5]    # 原码中的时空平面
            
            for grids in self.inr.encoder.grids:
                # ==================================
                # 1. PlaneTV (对应源码 PlaneTV 类)
                # ==================================
                # 源码逻辑：对所有平面算一遍 TV，再对空间平面单独累加一遍
                for grid_id in spatial_grids:
                    tv_plane += compute_plane_tv(grids[grid_id])
                for grid in grids:
                    tv_plane += compute_plane_tv(grid)
                    
                # ==================================
                # 2. TimeSmoothness (对应源码 TimeSmoothness 类)
                # ==================================
                # 源码逻辑：对包含时间的平面计算二阶导数平滑
                for grid_id in time_grids:
                    smooth_time += compute_plane_smoothness(grids[grid_id])
                    
                # ==================================
                # 3. L1TimePlanes (对应源码 L1TimePlanes 类)
                # ==================================
                # 源码逻辑：时间乘子应当尽可能保持为初始状态 1
                for grid_id in time_grids:
                    l1_time += torch.abs(1.0 - grids[grid_id]).mean()
                    
            return tv_plane, smooth_time, l1_time
    def deform_reg(self, out, xyz, e):
        """
        计算变形场的正则化损失，确保变形是平滑且合理的
        
        Args:
            out: 变形后的坐标
            xyz: 原始坐标
            e: 变形嵌入特征
            
        Returns:
            torch.Tensor: 变形正则化损失值
        """
        if True:  # 使用自动微分方法
            n_sample = 4  # 采样点数
            x = xyz[:, :n_sample].flatten(0, 1).detach()  # 采样坐标
            e = e[:, :n_sample].flatten(0, 1).detach()  # 采样嵌入特征

            x.requires_grad_()  # 启用梯度计算
            outputs = self.deform_net(x, e)  # 计算变形场
            
            # 计算雅可比矩阵
            grads = []
            out_sum = []
            for i in range(3):  # 对每个坐标分量
                out_sum.append(outputs[:, i].sum())  # 输出求和
                grads.append(
                    torch.autograd.grad((out_sum[-1],), (x,), create_graph=True)[0]
                )  # 计算梯度
            
            jacobian = torch.stack(grads, -1)  # 构建雅可比矩阵
            jtj = torch.matmul(jacobian, jacobian.transpose(-1, -2))  # J^T J
            I = torch.eye(3, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)  # 单位矩阵
            sq_residual = ((jtj - I) ** 2).sum((-2, -1))  # 计算残差平方和
            return torch.nan_to_num(sq_residual, 0.0, 0.0, 0.0).mean()  # 返回平均损失
        else:
            # 有限差分方法（备选）
            out = out - xyz  # 变形位移
            d_out2 = ((out - torch.flip(out, (1,))) ** 2).sum(-1) + 1e-6  # 位移差分平方
            dx2 = ((xyz - torch.flip(xyz, (1,))) ** 2).sum(-1) + 1e-6  # 坐标差分平方
            dd_dx = d_out2.sqrt() / dx2.sqrt()  # 近似变形梯度
            return F.smooth_l1_loss(dd_dx, torch.zeros_like(dd_dx).detach(), beta=1e-3)  # 平滑L1损失