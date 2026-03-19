import os
import argparse
import logging
import random
import numpy as np
import torch
from .pre_dcm_DYL import (
    load_cine_dicom_dataset,
    load_cine_dicom_dataset_with_source_registration,
)
from .inr.train import train
import json


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def apply_shorttrain_preset(args):
    """只覆盖最小必要超参，不改网络结构。"""
    presets = {
        "none": {},
        "2d_refine_2k": {
            "n_iter": 2000,
            "batch_size": 4096,
            "learning_rate": 0.003,
            "weight_tv": 0.001,
            "weight_smooth": 0.001,
            "weight_l1_time": 5e-5,
            "weight_bias": 0.05,
            "log_interval": 50,
            "eval_interval": 200,
            "nifti_interval": 2000,
        },
        "2d_refine_5k": {
            "n_iter": 5000,
            "batch_size": 4096,
            "learning_rate": 0.003,
            "weight_tv": 8e-4,
            "weight_smooth": 8e-4,
            "weight_l1_time": 5e-5,
            "weight_bias": 0.05,
            "log_interval": 50,
            "eval_interval": 500,
            "nifti_interval": 2500,
        },
    }
    selected = args.shorttrain_preset
    overrides = presets[selected]
    for k, v in overrides.items():
        setattr(args, k, v)
    return overrides


def setup_logger(output_dir):
    """配置全局日志，双写到终端和文件"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="🚀 4D Cardiac MRI Reconstruction Engine (K-Planes + NeSVoR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # =====================================================================
    # [模块 1]: 路径与系统环境 (I/O & Environment)
    # =====================================================================
    group_io = parser.add_argument_group("1. I/O & Environment (路径与环境)")
    group_io.add_argument("--dicom_dir", type=str, default="/data/dengyz/dataset/DYL/cine_dicom",
                          help="原始 DICOM 文件夹路径 (内含 cine_sax, cine_4ch 等 stack 文件夹)")
    group_io.add_argument("--output_dir", type=str, default="/data/dengyz/dataset/DYL/K_ne_output_v2/2",
                          help="所有输出文件、模型权重、NIfTI、日志的保存根目录")
    group_io.add_argument("--device", type=str, default="cuda",
                          help="计算设备 (默认 cuda)")
    group_io.add_argument(
        "--stacks",
        nargs="+",
        default=["cine_sax", "cine_4ch", "cine_2ch"],
        help="参与重建的 stack 名称列表（默认与原始实验一致：sax/4ch/2ch）",
    )
    group_io.add_argument(
        "--registration",
        type=str,
        default="stack",
        choices=["svort", "svort-stack", "svort-only", "stack", "none"],
        help="源码同款 registration 策略（建议先用 stack）",
    )
    group_io.add_argument(
        "--svort_version",
        type=str,
        default="v2",
        help="SVoRT 版本（仅当 registration 包含 svort 时生效）",
    )
    group_io.add_argument(
        "--scanner_space",
        action="store_true",
        help="registration 后强制映射回 scanner space（源码同款开关）",
    )
    group_io.add_argument("--num_workers", type=int, default=8,
                          help="DataLoader 的 CPU 线程数 (根据你的 CPU 核心数调整)")
    group_io.add_argument("--single_precision", action="store_true",
                          help="是否禁用 AMP 混合精度 (默认使用 FP16 加速，若 Loss 出现 NaN 可开启此项回退至 FP32)")
    group_io.add_argument("--debug", action="store_true",
                          help="开启调试模式 (会检查每一个梯度是否为 NaN，极度拖慢速度，仅供排错用)")

    # =====================================================================
    # [模块 2]: 物理采集与采样超参 (Physical & Sampling)
    # =====================================================================
    group_phys = parser.add_argument_group("2. Physical & Sampling (物理采样)")
    group_phys.add_argument("--psf_resolution", type=float, nargs=3, default=[1.25, 1.25, 6.0],
                            help="各向异性 PSF 采样的物理分辨率标准差 (dx, dy, dz)")
    group_phys.add_argument("--n_samples", type=int, default=9,
                            help="每条射线在层厚方向上的高斯采样点数 S")
    group_phys.add_argument("--chunk_size", type=int, default=200000,
                            help="生成 4D NIfTI 时的前向传播分块大小")

    # =====================================================================
    # [模块 3]: 网络架构参数
    # =====================================================================
    group_arch = parser.add_argument_group("3. Network Architecture (网络架构)")
    group_arch.add_argument("--width", type=int, default=64)
    group_arch.add_argument("--depth", type=int, default=4)
    group_arch.add_argument("--n_features_z", type=int, default=16)
    group_arch.add_argument("--n_features_slice", type=int, default=16)
    group_arch.add_argument("--n_levels_bias", type=int, default=2)
    group_arch.add_argument(
        "--fusion_mode",
        type=str,
        default="mul",
        choices=["mul", "grouped_mlp"],
        help="K-Planes plane fusion mode for P4 ablation"
    )
    group_arch.add_argument("--no_slice_scale", action="store_true")
    group_arch.add_argument("--no_slice_variance", action="store_true")
    group_arch.add_argument("--no_pixel_variance", action="store_true")
    group_arch.add_argument("--no_transformation_optimization", action="store_true")
    group_arch.add_argument("--deformable", action="store_true", default=False)
    group_arch.add_argument("--n_features_deform", type=int, default=8)
    group_arch.add_argument("--n_features_per_level_deform", type=int, default=4)
    group_arch.add_argument("--level_scale_deform", type=float, default=1.3819)
    group_arch.add_argument("--coarsest_resolution_deform", type=float, default=32.0)
    group_arch.add_argument("--finest_resolution_deform", type=float, default=8.0)
    group_arch.add_argument("--log2_hashmap_size", type=int, default=22)

    # =====================================================================
    # [模块 4]: 优化器与训练调度
    # =====================================================================
    group_opt = parser.add_argument_group("4. Optimization & Scheduling (优化器)")
    group_opt.add_argument("--n_iter", type=int, default=80000)
    group_opt.add_argument("--n_epochs", type=int, default=None)
    group_opt.add_argument("--batch_size", type=int, default=4096)
    group_opt.add_argument("--learning_rate", type=float, default=0.005)
    group_opt.add_argument(
        "--coarse_to_fine_enable",
        action="store_true",
        help="Enable coarse-to-fine scale activation schedule"
    )
    group_opt.add_argument(
        "--c2f_stage_iters",
        type=int,
        nargs=2,
        default=None,
        metavar=("S1", "S2"),
        help="Stage boundaries for coarse-to-fine: iter < S1, S1 <= iter < S2, iter >= S2"
    )
    group_opt.add_argument(
        "--shorttrain_preset",
        type=str,
        default="none",
        choices=["none", "2d_refine_2k", "2d_refine_5k"],
        help="短训验证预设（仅覆盖少量训练超参）"
    )

    # =====================================================================
    # [模块 5]: 正则化权重
    # =====================================================================
    group_reg = parser.add_argument_group("5. 4D Regularization (时空正则化权重)")
    group_reg.add_argument("--weight_tv", type=float, default=0.005)
    group_reg.add_argument("--weight_smooth", type=float, default=0.005)
    group_reg.add_argument("--weight_l1_time", type=float, default=0.0001)
    group_reg.add_argument("--weight_transformation", type=float, default=0.05)
    group_reg.add_argument("--weight_bias", type=float, default=0.1)
    group_reg.add_argument("--weight_deform", type=float, default=0.1)

    # =====================================================================
    # [模块 6]: 监控、日志与可视化
    # =====================================================================
    group_log = parser.add_argument_group("6. Logging & Hooks (监控与钩子)")
    group_log.add_argument("--log_interval", type=int, default=100)
    group_log.add_argument("--eval_interval", type=int, default=1000)
    group_log.add_argument("--nifti_interval", type=int, default=5000)
    group_log.add_argument("--seed", type=int, default=2026, help="全局随机种子")

    args = parser.parse_args()

    args.delta = 0.1
    args.n_features_per_level = 2
    args.dtype = torch.float32 if args.single_precision else torch.float16

    if not hasattr(args, 'weight_image'):
        args.weight_image = 1.0

    if args.deformable and not args.single_precision:
        logging.warning(
            "deformable + mixed precision 可能不稳定，建议加 --single_precision"
        )

    set_global_seed(args.seed)
    overrides = apply_shorttrain_preset(args)

    setup_logger(args.output_dir)
    logging.info("===================================================")
    logging.info("    🚀 4D Cardiac MRI Reconstruction Pipeline    ")
    logging.info("===================================================")

    config_dict = vars(args).copy()
    if 'dtype' in config_dict:
        config_dict['dtype'] = str(config_dict['dtype'])

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ All Hyperparameters successfully saved to: {config_path}")

    for arg, value in vars(args).items():
        logging.info(f"--{arg}: {value}")

    if overrides:
        logging.info("===================================================")
        logging.info(f"✅ shorttrain preset applied: {args.shorttrain_preset}")
        logging.info(f"✅ overridden args: {overrides}")

    logging.info("===================================================")

    logging.info("Step 1: Parsing DICOM files and building Affine matrices...")
    if args.registration != "none":
        logging.info(
            "Applying source-style registration: registration=%s, svort_version=%s",
            args.registration,
            args.svort_version,
        )
        images, affines, timestamps, masks = load_cine_dicom_dataset_with_source_registration(
            base_dir=args.dicom_dir,
            stack_names=args.stacks,
            registration=args.registration,
            device=args.device,
            svort_version=args.svort_version,
            scanner_space=args.scanner_space,
        )
    else:
        images, affines, timestamps, masks = load_cine_dicom_dataset(
            args.dicom_dir,
            stack_names=args.stacks,
        )

    if len(images) == 0:
        logging.error("No valid DICOM data loaded. Please check the DICOM directory.")
        return

    logging.info("Step 2: Igniting 4D K-Planes Engine...")
    train(
        images_list=images,
        affines_list=affines,
        timestamps_list=timestamps,
        masks_list=masks,
        args=args
    )

    logging.info("🎉 Training Completed Successfully!")


if __name__ == "__main__":
    main()
