import os
import time
import datetime
import logging
from argparse import Namespace
from typing import List, Tuple, Dict, Any, Optional
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import nibabel as nib

from ..utils import MovingAverage, log_params, TrainLogger
from .models import INR, NeSVoR, D_LOSS, S_LOSS, DS_LOSS, I_REG, B_REG, T_REG, D_REG
from ..transform import RigidTransform
from .data import DynamicMRIDataset  # 引入我们手写的 4D Dataset
#######已封存的v1版本的。


# ==========================================
# 监控与评估辅助函数 (Helper Functions)
# ==========================================

@torch.no_grad()
def render_and_evaluate(model, val_batch, spatial_scaling, center, save_dir, iteration, val_mask_2d):
    """渲染 2D 验证切片，计算 PSNR/SSIM，并保存真实图、预测图和误差热力图"""
    model.eval()
    
    xyzt = val_batch["xyzt"].to(model.args.device)
    v_target = val_batch["v"].to(model.args.device)
    slice_idx = val_batch["slice_idx"].to(model.args.device)
    
    # 【修复补充】：压缩切片索引
    slice_idx = slice_idx.squeeze(-1)

    # 空间坐标缩放对齐
    xyzt_scaled = xyzt.clone()
    
    se = None
    if model.args.n_features_slice:
        se = model.slice_embedding(slice_idx).unsqueeze(1)
        
    # 获取刚性位姿 (仅测试，无高斯扰动)
    t_pose = model.axisangle[slice_idx].unsqueeze(1)
    from nesvor.transform import ax_transform_points
    xyz_transformed = ax_transform_points(t_pose, xyzt_scaled[:, :3].unsqueeze(1), model.trans_first)
    
    xyzt_infer = torch.cat([xyz_transformed, xyzt_scaled[:, 3:].unsqueeze(1)], dim=-1)
    
    # 网络前向推理
    results = model.net_forward(xyzt_infer.view(-1, 4), se.view(-1, se.shape[-1]) if se is not None else None)
    
    density = results["density"]
    bias = results["log_bias"].exp() if "log_bias" in results else 1.0
    
    c = F.softmax(model.logit_coef, 0)[slice_idx] * model.n_slices if not model.args.no_slice_scale else 1.0
    v_pred = (bias * density).squeeze(-1) * c.squeeze(-1)
    
    mse = F.mse_loss(v_pred, v_target.squeeze(-1))
    psnr = -10.0 * torch.log10(mse + 1e-8)
    
# ==========================================
    # 【修复核心】：利用原始 Mask 还原 2D 画布
    # ==========================================
    if isinstance(val_mask_2d, torch.Tensor):
        mask_np = val_mask_2d.cpu().numpy().astype(bool)
    else:
        mask_np = np.array(val_mask_2d).astype(bool)
        
    H, W = mask_np.shape
    pred_img = np.zeros((H, W), dtype=np.float32)
    target_img = np.zeros((H, W), dtype=np.float32)
    
    # 按照 Mask 的 True 位置，将 1D 预测结果完美填回 2D 画布
    pred_img[mask_np] = v_pred.cpu().numpy()
    target_img[mask_np] = v_target.squeeze(-1).cpu().numpy()

    data_range = target_img.max() - target_img.min()
    ssim_val = ssim(target_img, pred_img, data_range=data_range)
    
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(target_img, cmap='gray')
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')
    
    axs[1].imshow(pred_img, cmap='gray')
    axs[1].set_title(f"Prediction (PSNR: {psnr:.2f})")
    axs[1].axis('off')
    
    error_map = np.abs(target_img - pred_img)
    im = axs[2].imshow(error_map, cmap='hot')
    axs[2].set_title(f"Error Map (SSIM: {ssim_val:.4f})")
    axs[2].axis('off')
    plt.colorbar(im, ax=axs[2])
    
    save_path = os.path.join(save_dir, f"render_iter_{iteration:05d}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    
    model.train()
    return psnr.item(), ssim_val


def plot_loss_curves(history_losses, save_dir, iteration):
    """绘制并保存所有的 Loss 曲线"""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for loss_name, loss_values in history_losses.items():
        if len(loss_values) > 0 and sum(loss_values) != 0:
            ax.plot(loss_values, label=loss_name, alpha=0.8)
            
    ax.set_yscale('log')
    ax.set_xlabel('Log Interval')
    ax.set_ylabel('Loss Value (Log Scale)')
    ax.set_title(f'Training Loss Curves @ Iter {iteration}')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig(os.path.join(save_dir, f"loss_curves_iter_{iteration:05d}.png"), bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def render_and_save_nifti(model, bb_4d, save_dir, iteration, spatial_res=(128, 128, 64), n_frames=8):
    """通过分块查询网络，生成纯净的 3D 和 4D 动态 NIfTI 图像，防止 OOM"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    min_xyz = bb_4d[0, :3]
    max_xyz = bb_4d[1, :3]
    
    x = torch.linspace(min_xyz[0], max_xyz[0], spatial_res[0], device=model.args.device)
    y = torch.linspace(min_xyz[1], max_xyz[1], spatial_res[1], device=model.args.device)
    z = torch.linspace(min_xyz[2], max_xyz[2], spatial_res[2], device=model.args.device)
    
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    xyz_flat = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)
    
    t_steps = torch.linspace(0, 1, n_frames + 1, device=model.args.device)[:-1]
    
    vol_4d = []
    chunk_size = getattr(model.args, 'chunk_size', 200000) 
    
    for t_val in t_steps:
        t_flat = torch.full((xyz_flat.shape[0], 1), t_val.item(), device=model.args.device)
        xyzt_flat = torch.cat([xyz_flat, t_flat], dim=-1)
        
        densities = []
        for i in range(0, xyzt_flat.shape[0], chunk_size):
            chunk = xyzt_flat[i:i + chunk_size]
            # =====================================================================
            # 【修复核心】：传入全 0 的 neutral embedding，补齐 tinycudann 缺失的维度
            # =====================================================================
            if model.args.n_features_slice > 0:
                se_neutral = torch.zeros((chunk.shape[0], model.args.n_features_slice), 
                                         dtype=chunk.dtype, device=chunk.device)
            else:
                se_neutral = None
                
            results = model.net_forward(chunk, se=se_neutral) 
            # =====================================================================
            density = results['density'].squeeze(-1)
            densities.append(density.cpu())
            
        vol_3d = torch.cat(densities, dim=0).view(spatial_res[0], spatial_res[1], spatial_res[2]).numpy()
        vol_4d.append(vol_3d)
        
    vol_4d_np = np.stack(vol_4d, axis=-1).astype(np.float32)
    affine = np.eye(4)
    
    img_3d = nib.Nifti1Image(vol_4d_np[..., 0], affine)
    nib.save(img_3d, os.path.join(save_dir, f"recon_3d_iter_{iteration:05d}.nii.gz"))
    
    img_4d = nib.Nifti1Image(vol_4d_np, affine)
    nib.save(img_4d, os.path.join(save_dir, f"recon_4d_cine_iter_{iteration:05d}.nii.gz"))
    
    del xyzt_flat, xyz_flat, grid_x, grid_y, grid_z
    torch.cuda.empty_cache()
    model.train()


# ==========================================
# 主训练管道 (Main Training Pipeline)
# ==========================================

def train(
    images_list: list, 
    affines_list: list, 
    timestamps_list: list, 
    masks_list: list, 
    args: Namespace
) -> Tuple[INR, Optional[list], Optional[torch.Tensor]]:
    
    dataset = DynamicMRIDataset(
        images_list=images_list,
        affines_list=affines_list,
        timestamps_list=timestamps_list,
        masks_list=masks_list,
        psf_resolution=getattr(args, 'psf_resolution', (1.0, 1.0, 5.0)) 
    )
    # =====================================================================
    # 【新增修复】：将 DICOM 的巨大像素值归一化到 [0, 1]，防止方差网络摆烂导致梯度消失！
    # =====================================================================
    valid_v = dataset.v[dataset.v > 0] # 提取有效像素
    if len(valid_v) > 0:
        # 突破 PyTorch quantile 的 1677 万元素限制：均匀采样计算分位数
        max_elements = 10000000 # 安全阈值设为 1000 万
        if len(valid_v) > max_elements:
            step = len(valid_v) // max_elements + 1
            sample_v = valid_v[::step] # 均匀切片采样
        else:
            sample_v = valid_v
            
        # 取 99% 分位数作为最大值，过滤掉个别血管或伪影的极亮噪点
        v_max = torch.quantile(sample_v.float(), 0.99) 
        dataset.v = torch.clamp(dataset.v / v_max, 0.0, 1.0)
    # =====================================================================
    
    if args.n_epochs is not None:
        args.n_iter = args.n_epochs * (dataset.v.numel() // args.batch_size)

    use_scaling = True
    use_centering = True
    spatial_scaling = 30.0 if use_scaling else 1.0

    # 时空绝缘缩放
    xyz_min = dataset.xyzt[:, :3].min(dim=0)[0]
    xyz_max = dataset.xyzt[:, :3].max(dim=0)[0]
    center = (xyz_min + xyz_max) / 2.0 if use_centering else torch.zeros_like(xyz_min)
    dataset.xyzt[:, :3] = (dataset.xyzt[:, :3] - center) / spatial_scaling
    
    new_xyz_min = (xyz_min - center) / spatial_scaling
    new_xyz_max = (xyz_max - center) / spatial_scaling
    t_min = torch.tensor([0.0], device=new_xyz_min.device)
    t_max = torch.tensor([1.0], device=new_xyz_max.device)
    
    bb_4d_min = torch.cat([new_xyz_min, t_min], dim=-1)
    bb_4d_max = torch.cat([new_xyz_max, t_max], dim=-1)
    bb_4d = torch.stack([bb_4d_min, bb_4d_max], dim=0)

    bb_4d_unscaled_min = torch.cat([xyz_min, t_min], dim=-1)
    bb_4d_unscaled_max = torch.cat([xyz_max, t_max], dim=-1)
    bb_4d_unscaled = torch.stack([bb_4d_unscaled_min, bb_4d_unscaled_max], dim=0)

    n_slices = len(images_list)
    initial_transformation = RigidTransform(torch.zeros(n_slices, 6))
    
# 【修复 1】：将全局 [3] 的分辨率张量，复制扩展成 [N_slices, 3]，满足底层查表要求
    psf_res_tensor = torch.tensor(getattr(args, 'psf_resolution', (1.25, 1.5, 6.0)), dtype=torch.float32)
    resolution_tensor = psf_res_tensor.unsqueeze(0).expand(n_slices, 3) / spatial_scaling
    
    model = NeSVoR(
        transformation=initial_transformation,
        resolution=resolution_tensor, # <--- 传入修复好的 [N_slices, 3] 矩阵
        v_mean=dataset.v.mean().item(),
        bounding_box=bb_4d,
        spatial_scaling=spatial_scaling,
        args=args,
    )

    # 优化器自动分组
    params_net = []
    params_encoding = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            if "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)
                
    logging.debug(log_params(model))
    optimizer = torch.optim.AdamW(
        params=[
            {"name": "encoding", "params": params_encoding},
            {"name": "net", "params": params_net, "weight_decay": 1e-2},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    # # milestone式衰减
    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer=optimizer,
    #     milestones=list(range(1, len(args.milestones) + 1)),
    #     gamma=args.gamma,
    # )
    # decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    #余弦退火学习率衰减
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.n_iter,
        eta_min=args.learning_rate * 0.01,
    )

    fp16 = not getattr(args, "single_precision", False)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0, enabled=fp16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000
    )

    model.train()
    loss_weights = {
        D_LOSS: 1.0,
        S_LOSS: 1.0,
        T_REG: args.weight_transformation,
        B_REG: getattr(args, 'weight_bias', 0.0),
        I_REG: getattr(args, 'weight_image', 1.0), # 时空正则化包
        D_REG: getattr(args, 'weight_deform', 0.0),
    }
    average = MovingAverage(1 - 0.001)

    # 初始化 Loss 记录器
    history_losses = {}
    log_interval = getattr(args, 'log_interval', 100)

    # DataLoader
    num_workers = getattr(args, 'num_workers', 0)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    data_iterator = iter(dataloader)

    logging_header = False
    logging.info("4D K-Planes NeSVoR Pipeline Training Starts.")
    train_time = 0.0

    # 预准备验证数据（取切片 0 用于快速 2D 监控）
    val_mask = (dataset.slice_idx.squeeze() == 0)
    val_batch = {
        "xyzt": dataset.xyzt[val_mask],
        "v": dataset.v[val_mask],
        "slice_idx": dataset.slice_idx[val_mask],
        "slice_R": dataset.slice_R[val_mask]
    }
    # 【新增】获取第0张切片的原始 2D 形状掩膜
    val_mask_2d = masks_list[0]

    output_dir = getattr(args, 'output_dir', './output')

    # 创建动态进度条，动态计算 ETA 和速度
    pbar = tqdm(range(1, args.n_iter + 1), desc="🚀 4D Recon", dynamic_ncols=True)
    
    for i in pbar:  # <--- 使用 pbar 替代原来的 range
        train_step_start = time.time()
        
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)
            
        batch = {k: v.to(args.device) for k, v in batch.items()}

        with torch.amp.autocast('cuda', enabled=fp16):
            losses = model(**batch)
            loss = 0.0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    loss = loss + loss_weights[k] * losses[k]
                    
        scaler.scale(loss).backward()
        
        if args.debug:
            for _name, _p in model.named_parameters():
                if _p.grad is not None and not _p.grad.isfinite().all():
                    logging.warning("iter %d: Found NaNs in the grad of %s", i, _name)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        train_time += time.time() - train_step_start
        
        for k in losses:
            average(k, losses[k].item())
            
        if i % log_interval == 0:
            for k in losses:
                if k not in history_losses:
                    history_losses[k] = [] # 如果遇到新的 Loss Key，自动帮它建个列表
                history_losses[k].append(losses[k].item())

            # 【新增】：将关键指标实时刷新到进度条尾部
            # ==========================================
            pbar.set_postfix({
                "D_Loss": f"{losses.get(D_LOSS, 0):.4f}", 
                "TV": f"{losses.get('TV_Plane', losses.get(I_REG, 0)):.4f}"
            })

        # ==========================================
        # 钩子 1: 2D 快速验证 (每 1000 步)
        # ==========================================
        eval_interval = getattr(args, 'eval_interval', 1000)
        if i % eval_interval == 0 or i == args.n_iter:
            save_dir = os.path.join(output_dir, "vis_2d_progress")
            psnr_val, ssim_val = render_and_evaluate(model, val_batch, spatial_scaling, center, save_dir, i, val_mask_2d)
            logging.info(f"[Validation @ Iter {i}] PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

            # 【新增】：将量化指标追加写入 CSV 文件
            metrics_file = os.path.join(output_dir, "metrics.csv")
            file_exists = os.path.isfile(metrics_file)
            with open(metrics_file, "a") as f:
                if not file_exists:
                    f.write("iteration,psnr,ssim\n") # 写入表头
                f.write(f"{i},{psnr_val:.4f},{ssim_val:.4f}\n")

        # ==========================================
        # 钩子 2: 3D/4D NIfTI 保存与 Loss 曲线 (每 5000 步)
        # ==========================================
        nifti_interval = getattr(args, 'nifti_interval', 5000)
        if i % nifti_interval == 0 or i == args.n_iter:
            progress_dir = os.path.join(output_dir, "training_progress")
            plot_loss_curves(history_losses, progress_dir, i)
            
            logging.info(f"Rendering 4D NIfTI volume at iteration {i}... This may take a moment.")
            render_and_save_nifti(
                model=model, bb_4d=bb_4d, save_dir=progress_dir, iteration=i,
                spatial_res=(128, 128, 64), n_frames=8
            )
            logging.info(f"NIfTI saved to {progress_dir}.")
        # 【新增】：保存模型权重与优化器状态 (支持断点续训或后续推理)
            ckpt_path = os.path.join(progress_dir, f"model_iter_{i:05d}.pt")
            torch.save({
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'spatial_scaling': spatial_scaling,
                'center': center
            }, ckpt_path)
            logging.info(f"Model checkpoint saved to {ckpt_path}")
        # 学习率调度与日志打印
        if i < args.n_iter:
            scheduler.step()

        # 2. 规律性打印详细训练日志表格 (这里设为每 eval_interval 步打印一次，避免刷屏)
        eval_interval = getattr(args, 'eval_interval', 1000)
        if i % eval_interval == 0 or i == args.n_iter:
            if not logging_header:
                train_logger = TrainLogger("time", "epoch", "iter", *list(losses.keys()), "lr")
                logging_header = True
            
            current_epoch = i // len(dataloader)
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)), current_epoch, i,
                *[average[k] for k in losses], optimizer.param_groups[0]["lr"],
            )
                
            if scaler.is_enabled():
                current_scaler = scaler.get_scale()
                if current_scaler < 1 / (2**5):
                    logging.warning("Numerical instability detected! GradScaler too small.")
    # ==========================================
    # 训练后处理
    # ==========================================
    transformation = model.transformation
    ax = transformation.axisangle()
    ax[:, -3:] *= spatial_scaling
    transformation = RigidTransform(ax)
    transformation = RigidTransform(torch.cat([torch.zeros_like(center), center])[None]).compose(transformation)
    
    model.inr.bounding_box.copy_(bb_4d_unscaled)
    dataset.xyzt[:, :3] *= spatial_scaling
    dataset.xyzt[:, :3] += center

    return model.inr, None, None