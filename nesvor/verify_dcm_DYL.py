import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pre_dcm_DYL import load_cine_dicom_dataset

def run_verification(dicom_dir):
    print("开始加载数据进行校验...")
    images, affines, timestamps, masks = load_cine_dicom_dataset(dicom_dir)
    total_slices = len(images)
    print(f"数据加载完毕，共 {total_slices} 张 2D 切片。")

    # ==========================================
    # 检验 1: 量化信息 Sanity Check
    # ==========================================
    unique_t = sorted(list(set(timestamps)))
    print(f"\n[量化检验] 发现 {len(unique_t)} 个独立的心跳相位 (t值):")
    print([round(t, 3) for t in unique_t])
    
    # 检查 Affine 的行列式 (应约等于像素面积 dx * dy)
    det = np.linalg.det(affines[0][:3, :3])
    print(f"[量化检验] Affine 矩阵行列式 (像素物理面积): {det:.4f} mm^2")

    # ==========================================
    # 检验 2: Mask 准确性与 2D 图像检查
    # ==========================================
    test_idx = total_slices // 2
    img = images[test_idx]
    mask = masks[test_idx]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Raw Image (idx={test_idx}, t={timestamps[test_idx]:.2f})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='Greens', alpha=0.3) # 绿色半透明叠加
    plt.title("Image + Background Mask Overlay")
    plt.axis('off')
    plt.savefig("check_mask.png")
    print("\n[可视化] Mask 叠加图已保存为 check_mask.png")

    # ==========================================
    # 检验 3: 3D 空间正交穿插测试 (极其核弹级的校验)
    # ==========================================
    # 策略：在 t=0 的所有切片中，寻找法线方向差异最大的 3 张切片（通常对应 SAX, 2CH, 4CH）
    t0_indices = [i for i, t in enumerate(timestamps) if t == unique_t[0]]
    
    if len(t0_indices) >= 3:
        # 取前三张差异最大的切片
        # 因为数据是按 stack 读取的，我们直接按列表长度的三等分取样，大概率能取到不同的 stack
        idx_sax = t0_indices[0]
        idx_4ch = t0_indices[len(t0_indices) // 3]
        idx_2ch = t0_indices[2 * len(t0_indices) // 3]
        
        probe_indices = [idx_sax, idx_4ch, idx_2ch]
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for p_idx in probe_indices:
            img_p = images[p_idx]
            aff_p = affines[p_idx]
            H, W = img_p.shape
            
            # 为了 3D 绘图速度，对图像进行降采样
            step = 4 
            grid_x, grid_y = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
            grid_z = np.zeros_like(grid_x)
            ones = np.ones_like(grid_x)
            
            # 展平并映射到 3D 世界坐标
            local_coords = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), ones.flatten()], axis=0)
            world_coords = aff_p @ local_coords
            
            X = world_coords[0, :].reshape(grid_x.shape)
            Y = world_coords[1, :].reshape(grid_y.shape)
            Z = world_coords[2, :].reshape(grid_z.shape)
            
            # 归一化图像亮度用于 3D 颜色映射
            img_surface = img_p[::step, ::step]
            img_surface = (img_surface - img_surface.min()) / (img_surface.max() - img_surface.min() + 1e-8)
            
            ax.plot_surface(X, Y, Z, facecolors=plt.cm.gray(img_surface), rstride=1, cstride=1, alpha=0.8, shade=False)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title("3D Affine Intersection Check (SAX, 4CH, 2CH)")
        plt.savefig("check_3d_intersection.png")
        print("[可视化] 3D 空间切片穿插图已保存为 check_3d_intersection.png")

    # ==========================================
    # 检验 4: 时序连贯性提取
    # ==========================================
    # 找寻空间位置完全一致（Affine 近似相等），但 t 不同的切片，看是否构成一个完整心跳
    ref_affine = affines[0]
    cine_sequence = []
    
    for i in range(total_slices):
        # 简单判断空间位置是否一致 (平移向量的距离差 < 1mm)
        if np.linalg.norm(affines[i][:3, 3] - ref_affine[:3, 3]) < 1.0:
            cine_sequence.append((timestamps[i], images[i]))
            
    cine_sequence.sort(key=lambda x: x[0]) # 按时间戳 t 排序
    
    if len(cine_sequence) > 1:
        plt.figure(figsize=(15, 3))
        num_frames_to_plot = min(10, len(cine_sequence))
        indices_to_plot = np.linspace(0, len(cine_sequence)-1, num_frames_to_plot, dtype=int)
        
        for j, seq_idx in enumerate(indices_to_plot):
            t_val, img_frame = cine_sequence[seq_idx]
            plt.subplot(1, num_frames_to_plot, j+1)
            plt.imshow(img_frame, cmap='gray')
            plt.title(f"t={t_val:.2f}")
            plt.axis('off')
            
        plt.suptitle("Cine Sequence Time Continuity Check")
        plt.savefig("check_time_continuity.png")
        print("[可视化] 时序连贯性抽帧图已保存为 check_time_continuity.png")
        
    print("\n校验完成！请务必仔细检查生成的三张 .png 图片。")

if __name__ == "__main__":
    # 请替换为你的实际 DICOM 路径
    TEST_DICOM_DIR = "/data/dengyz/dataset/DYL/cine_dicom" 
    run_verification(TEST_DICOM_DIR)