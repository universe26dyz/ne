import os
import glob
import pydicom
import numpy as np
import torch
import logging
#######已封存的v1版本的。
def compute_affine_from_dicom(ds):
    """
    根据 DICOM 头文件计算 2D 切片到物理世界坐标的 4x4 仿射矩阵
    """
    # 提取空间朝向 (Row Cosines 和 Column Cosines)
    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
    row_cosine = iop[0:3]
    col_cosine = iop[3:6]
    
    # 提取物理位置 (图像左上角的绝对坐标)
    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
    
    # 提取面内分辨率 (Pixel Spacing 通常是 [Row Spacing, Column Spacing] = [dy, dx])
    # 注意：DICOM 坐标系中，row 对应 y (上下)，col 对应 x (左右)
    if 'PixelSpacing' in ds:
        ps = np.array([float(x) for x in ds.PixelSpacing])
        dy, dx = ps[0], ps[1]
    else:
        # 极少部分数据可能缺省，默认给 1.0mm
        dy, dx = 1.0, 1.0 

    # 计算法线向量 (Z轴方向)
    normal = np.cross(row_cosine, col_cosine)
    
    # 构建 4x4 刚体仿射矩阵
    affine = np.eye(4)
    # X 轴向量 (列方向)
    affine[:3, 0] = col_cosine * dx
    # Y 轴向量 (行方向)
    affine[:3, 1] = row_cosine * dy
    # Z 轴向量 (切片法线方向，用于后续 PSF 模拟层厚)
    affine[:3, 2] = normal
    # 平移向量 (原点绝对物理位置)
    affine[:3, 3] = ipp
    
    return affine

def generate_background_mask(image_array, threshold_ratio=0.05):
    """
    简单的阈值掩膜生成，剔除空气背景
    """
    max_val = image_array.max()
    mask = image_array > (max_val * threshold_ratio)
    return mask

def load_cine_dicom_dataset(base_dir):
    """
    解析整个 Cine DICOM 文件夹，返回四维网络所需的弹药库
    """
    stacks = ['cine_sax', 'cine_4ch', 'cine_3ch', 'cine_2ch']
    
    images_list = []
    affines_list = []
    timestamps_list = []
    masks_list = []
    
    logging.info(f"Scanning DICOM directory: {base_dir}")
    
    for stack_name in stacks:
        stack_dir = os.path.join(base_dir, stack_name)
        if not os.path.exists(stack_dir):
            logging.warning(f"Stack directory not found, skipping: {stack_dir}")
            continue
            
        dcm_files = glob.glob(os.path.join(stack_dir, '**', '*.dcm'), recursive=True)
        logging.info(f"Found {len(dcm_files)} DICOM files in {stack_name}.")
        
        # 1. 第一遍扫描：收集当前 stack 的所有唯一 TriggerTime 以推导时间 t
        trigger_times = []
        valid_datasets = []
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                if 'TriggerTime' in ds:
                    tt = float(ds.TriggerTime)
                    trigger_times.append(tt)
                    valid_datasets.append((f, ds))
            except Exception as e:
                continue
                
        if not valid_datasets:
            continue
            
        # 2. 对 TriggerTime 进行聚类/去重排序，计算归一化心动周期 t
        # 使用 round(tt, 1) 处理同一相位可能存在几毫秒的浮点误差
        unique_phases = sorted(list(set([round(t, 1) for t in trigger_times])))
        num_phases = len(unique_phases)
        logging.info(f"Detected {num_phases} distinct cardiac phases in {stack_name}.")
        
        # 3. 第二遍扫描：读取像素，组装最终列表
        for f, ds in valid_datasets:
            ds_full = pydicom.dcmread(f)
            
            # 读取像素矩阵 (转换为 float32)
            img = ds_full.pixel_array.astype(np.float32)
            
            # 计算局部仿射矩阵
            affine = compute_affine_from_dicom(ds_full)
            
            # 映射时间戳 t 到 [0, 1)
            current_phase_rounded = round(float(ds_full.TriggerTime), 1)
            phase_idx = unique_phases.index(current_phase_rounded)
            t_val = phase_idx / num_phases  # 例如 0, 0.04, 0.08...
            
            # 生成 Mask
            mask = generate_background_mask(img)
            
            images_list.append(img)
            affines_list.append(affine)
            timestamps_list.append(t_val)
            masks_list.append(mask)
            
    logging.info(f"Dataset fully loaded! Total 2D slices: {len(images_list)}")
    return images_list, affines_list, timestamps_list, masks_list