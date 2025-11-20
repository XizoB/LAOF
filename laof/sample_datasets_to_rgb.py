#!/usr/bin/env python3

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2

def visualize_demo_data(file_path, demo_name, output_dir="./visualization"):
    """
    可视化demo数据，将obs、obs_of、obs_sam在同一行显示并保存为MP4
    
    Args:
        file_path: HDF5文件路径
        demo_name: demo名称
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(file_path, 'r') as f:
        if demo_name not in f:
            print(f"Demo {demo_name} 不存在于文件中")
            return
        
        demo_group = f[demo_name]
        
        # 读取数据
        obs = demo_group['obs'][:]           # (T, 64, 64, 3)
        obs_of = demo_group['obs_of'][:]     # (T, 64, 64, 2) 
        obs_sam = demo_group['obs_sam'][:]   # (T, 64, 64)
        
        T = obs.shape[0]
        print(f"可视化 {demo_name}: {T} 帧")
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{demo_name} Visualization', fontsize=16)
        
        # 初始化显示
        im1 = axes[0].imshow(obs[0], animated=True)
        axes[0].set_title('Original Observation (obs)')
        axes[0].axis('off')
        
        # 光流可视化 - 将2通道光流转换为RGB
        flow_rgb = flow_to_rgb(obs_of[0])
        im2 = axes[1].imshow(flow_rgb, animated=True)
        axes[1].set_title('Optical Flow (obs_of)')
        axes[1].axis('off')
        
        # SAM mask可视化
        flow_sam_rgb = flow_to_rgb(obs_sam[0])
        im3 = axes[2].imshow(flow_sam_rgb, animated=True)
        axes[2].set_title('SAM Segmentation (obs_sam)')
        axes[2].axis('off')

        # im3 = axes[2].imshow(obs_sam[0], cmap='gray', animated=True)
        # axes[2].set_title('SAM Segmentation (obs_sam)')
        # axes[2].axis('off')
        
        plt.tight_layout()
        
        def animate(frame):
            # 更新obs
            im1.set_array(obs[frame])
            
            # 更新obs_of
            flow_rgb = flow_to_rgb(obs_of[frame])
            im2.set_array(flow_rgb)
            
            # 更新obs_sam
            flow_sam_rgb = flow_to_rgb(obs_sam[frame])
            im3.set_array(flow_sam_rgb)
            
            # im3.set_array(obs_sam[frame])
            
            return [im1, im2, im3]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=T, interval=100, blit=True, repeat=True)
        
        # 保存为MP4
        output_path_mp4 = os.path.join(output_dir, f"{demo_name}_visualization.mp4")
        writer_mp4 = animation.FFMpegWriter(fps=10, metadata=dict(artist='Demo Visualization'), bitrate=1800)
        
        # 保存为GIF
        output_path_gif = os.path.join(output_dir, f"{demo_name}_visualization.gif")
        writer_gif = animation.PillowWriter(fps=5, metadata=dict(artist='Demo Visualization'), bitrate=1800)
        
        try:
            print(f"正在保存MP4视频...")
            anim.save(output_path_mp4, writer=writer_mp4)
            print(f"✓ MP4视频已保存: {output_path_mp4}")
        except Exception as e:
            print(f"✗ MP4保存失败: {e}")
        
        try:
            print(f"正在保存GIF动画...")
            anim.save(output_path_gif, writer=writer_gif)
            print(f"✓ GIF动画已保存: {output_path_gif}")
        except Exception as e:
            print(f"✗ GIF保存失败: {e}")
        
        plt.close(fig)


def flow_to_rgb(flow):
    """
    将光流数据转换为RGB图像用于可视化
    
    Args:
        flow: numpy array (H, W, 2) 光流数据
    
    Returns:
        rgb: numpy array (H, W, 3) RGB图像
    """
    h, w = flow.shape[:2]
    
    # 计算光流的幅度和角度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 创建HSV图像
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 色调代表角度
    hsv[..., 0] = angle * 180 / np.pi / 2
    
    # 饱和度设为最大
    hsv[..., 1] = 255
    
    # 亮度代表幅度
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    # 转换为RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def test_hdf5_structure(file_path):
    """
    测试读取包含光流和LangSAM数据的HDF5文件
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    print(f"读取HDF5文件: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"文件中的demo数量: {len(f.keys())}")
        
        # 遍历所有demo
        for demo_name in f.keys():
            print(f"\n=== {demo_name} ===")
            demo_group = f[demo_name]
            
            # 显示每个数据集的信息
            expected_keys = ['obs', 'ta', 'done', 'rewards', 'ep_returns', 'obs_of', 'obs_sam']
            
            for key in expected_keys:
                if key in demo_group.keys():
                    dataset = demo_group[key]
                    print(f"  ✓ {key}.shape: {dataset.shape} {dataset.dtype}")
                    
                    # 显示一些统计信息
                    if key == 'ep_returns':
                        print(f"    ep_returns unique values: {np.unique(dataset[:])}")
                    elif key == 'rewards':
                        data = dataset[:]
                        print(f"    rewards sum: {np.sum(data):.2f}")
                    elif key == 'done':
                        data = dataset[:]
                        print(f"    done count: {np.sum(data)}")
                    elif key == 'obs_of':
                        data = dataset[:]
                        print(f"    obs_of range: [{np.min(data):.3f}, {np.max(data):.3f}]")
                    elif key == 'obs_sam':
                        data = dataset[:]
                        print(f"    obs_sam range: [{np.min(data):.3f}, {np.max(data):.3f}]")
                else:
                    print(f"  ✗ {key}: 缺失")
        
        # 可视化第一个demo
        first_demo = list(f.keys())[0]
        print(f"\n开始可视化 {first_demo}...")
        
    # 调用可视化函数
    visualize_demo_data(file_path, first_demo)

if __name__ == "__main__":
    # 测试读取数据集
    save_dir = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/trian"
    save_dir = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/chaser/train"
    save_dir = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/bigfish1/train"

    
    # 查找所有hdf5文件
    for i in range(10):  # 检查前10个可能的文件
        file_path = os.path.join(save_dir, f"{i}.hdf5")
        if os.path.exists(file_path):
            test_hdf5_structure(file_path)
            print(f"\n可视化视频已保存到 ./visualization/ 目录")
            break
    else:
        print(f"在目录 {save_dir} 中未找到HDF5文件")
        print("请先运行采样程序生成数据文件")
