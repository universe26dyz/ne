import os
import argparse
import logging
import torch
from pre_dcm_DYL import load_cine_dicom_dataset
from nesvor.inr.train import train
import json
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
    group_io.add_argument("--output_dir", type=str, default="/data/dengyz/dataset/DYL/K_ne_output/8", 
                          help="所有输出文件、模型权重、NIfTI、日志的保存根目录")
    group_io.add_argument("--device", type=str, default="cuda", 
                          help="计算设备 (默认 cuda)")
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
    group_phys.add_argument("--psf_resolution", type=float, nargs=3, default=[1.25, 1.5, 6.0], 
                            help="各向异性 PSF 采样的物理分辨率标准差 (dx, dy, dz)，通常 dz (层厚) 远大于 dx/dy,要严格与dicom中的分辨率一致")
    group_phys.add_argument("--n_samples", type=int, default=9, 
                            help="每条射线（即每个像素点）在层厚方向上的高斯采样点数 S")
    group_phys.add_argument("--chunk_size", type=int, default=200000, 
                            help="生成 4D NIfTI 时的前向传播分块大小 (防 OOM，12G 显存建议 10w，24G 建议 25w)")

    # =====================================================================
    # [模块 3]: 网络架构参数 (Network Architecture)
    # =====================================================================
    group_arch = parser.add_argument_group("3. Network Architecture (网络架构)")
    # 注意：K-Planes 的特征维度已在 models.py 初始化时硬编码 (如 feature_dim=32)，这里主要是 NeSVoR 下游 MLP 的参数
    group_arch.add_argument("--width", type=int, default=64, 
                            help="下游 MLP (Density, Sigma, Bias) 的隐藏层神经元宽度")
    group_arch.add_argument("--depth", type=int, default=4, 
                            help="下游 MLP (Density) 的隐藏层深度")
    group_arch.add_argument("--n_features_z", type=int, default=16, 
                            help="Density MLP 吐给 Sigma MLP 的中间特征维度")
    group_arch.add_argument("--n_features_slice", type=int, default=16, 
                            help="切片独立特征 (Slice Embedding) 的维度")
    group_arch.add_argument("--n_levels_bias", type=int, default=2, 
                            help="Bias Field 网络使用的特征缩放层级 (值越大，偏置场越复杂)")
    group_arch.add_argument("--no_slice_scale", action="store_true", help="禁用切片级别的亮度缩放自适应")
    group_arch.add_argument("--no_slice_variance", action="store_true", help="禁用切片级别的方差预测")
    group_arch.add_argument("--no_pixel_variance", action="store_true", help="禁用像素级别的运动方差预测 (Sigma Net)")
    group_arch.add_argument("--no_transformation_optimization", action="store_true", help="冻结切片刚性位姿 (Axisangle)，不进行优化")
    group_arch.add_argument("--deformable", action="store_false", default=False, 
                            help="【4D慎用】开启非刚性 DeformNet (默认 False，极易与心脏跳动时间流发生形变歧义)")

    # =====================================================================
    # [模块 4]: 优化器与训练调度 (Optimization & Scheduling)
    # =====================================================================
    group_opt = parser.add_argument_group("4. Optimization & Scheduling (优化器)")
    group_opt.add_argument("--n_iter", type=int, default=80000, 
                           help="训练总迭代次数 (通常 4D 场景需要 3w~8w 步)")
    group_opt.add_argument("--n_epochs", type=int, default=None, 
                           help="按 Epoch 指定训练时长 (如果设置，将覆盖 n_iter)")
    group_opt.add_argument("--batch_size", type=int, default=4096, 
                           help="Batch Size (每次迭代送入的像素射线数量)")
    group_opt.add_argument("--learning_rate", type=float, default=0.005, 
                           help="初始学习率")
    # group_opt.add_argument("--milestones", type=float, nargs='+', default=[0.3,0.5, 0.75], 
                        #    help="学习率衰减节点 (按总 n_iter 的比例，默认在 50% 和 75% 时衰减)")
    # group_opt.add_argument("--gamma", type=float, default=0.5, 
    #                        help="学习率衰减系数")

    # =====================================================================
    # [模块 5]: 4D 时空正则化权重 (4D Regularization Weights) -> 调参核心区域！
    # =====================================================================
    group_reg = parser.add_argument_group("5. 4D Regularization (时空正则化权重)")
    group_reg.add_argument("--weight_tv", type=float, default=0.005, 
                           help="K-Planes 空间全变分 (TV_Plane) 权重。太大则解剖模糊，太小则边缘毛刺")
    group_reg.add_argument("--weight_smooth", type=float, default=0.005, 
                           help="K-Planes 时间二阶平滑 (Smooth_Time) 权重。防抽搐核心，通常比空间 TV 大一个量级")
    group_reg.add_argument("--weight_l1_time", type=float, default=0.0001, 
                           help="K-Planes 时间 L1 稀疏 (L1_Time) 权重。促使背景区域的时间乘子保持为 1 (静止)")
    group_reg.add_argument("--weight_transformation", type=float, default=0.05, 
                           help="切片刚性位姿的 L2 正则化权重 (防止切片飘得太远)")
    group_reg.add_argument("--weight_bias", type=float, default=0.1, 
                           help="偏置场的 L2 正则化权重 (防止网络用 Bias 疯狂补齐解剖细节)")
    group_reg.add_argument("--weight_deform", type=float, default=0.1, 
                           help="DeformNet 的平滑正则化权重 (仅在 deformable=True 时生效)")

    # =====================================================================
    # [模块 6]: 监控、日志与可视化钩子 (Logging & Hooks)
    # =====================================================================
    group_log = parser.add_argument_group("6. Logging & Hooks (监控与钩子)")
    group_log.add_argument("--log_interval", type=int, default=100, 
                           help="终端打印与收集 Loss 曲线数据的频率 (迭代步数)")
    group_log.add_argument("--eval_interval", type=int, default=1000, 
                           help="进行 2D 纯净切片渲染与 PSNR/SSIM 计算的频率")
    group_log.add_argument("--nifti_interval", type=int, default=5000, 
                           help="进行 3D/4D 完整 NIfTI 容积生成的频率 (比较耗时，不建议太小)")

    args = parser.parse_args()
    # [新增] 手动补齐 NeSVoR 底层依赖的隐式超参数 (防止底层代码找不到属性报错)
    # =====================================================================
    args.delta = 0.1                  # 原版 NeSVoR 边缘正则化系数，初始化仍需调用
    args.n_features_per_level = 2     # K-Planes 每个网格层级的特征维度基数
    args.dtype = torch.float32 if args.single_precision else torch.float16 # 网络推理数据类型
    
    # 兼容 train.py 字典所需的占位权重
    if not hasattr(args, 'weight_image'):
        args.weight_image = 1.0       # 已经被 K-Planes 的三项正则化接管，此处作为外壳开关即可

    # 初始化日志系统
    setup_logger(args.output_dir)
    logging.info("===================================================")
    logging.info("    🚀 4D Cardiac MRI Reconstruction Pipeline    ")
    logging.info("===================================================")
    
    # =====================================================================
    # 【新增】：将所有超参数完整保存为 config.json，用于后续完美复刻
    # =====================================================================
    config_dict = vars(args).copy()
    # 特殊处理：torch.dtype 对象无法被 json 序列化，必须转为字符串
    if 'dtype' in config_dict:
        config_dict['dtype'] = str(config_dict['dtype'])
        
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ All Hyperparameters successfully saved to: {config_path}")
    # =====================================================================

    # 打印所有的参数配置供复查
    for arg, value in vars(args).items():
        logging.info(f"--{arg}: {value}")
        
    logging.info("===================================================")

    # 1. 调用数据装填器
    logging.info("Step 1: Parsing DICOM files and building Affine matrices...")
    images, affines, timestamps, masks = load_cine_dicom_dataset(args.dicom_dir)
    
    if len(images) == 0:
        logging.error("No valid DICOM data loaded. Please check the DICOM directory.")
        return

    # 2. 点火！启动 4D 训练引擎
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