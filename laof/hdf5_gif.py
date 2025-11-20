#!/usr/bin/env python3

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2
from PIL import Image

def save_demo_images(file_path, demo_name, output_base_dir="/root/Docker_RoboLearn/Benchmark/LAPO_IRL/Test/videos"):
    """
    将demo数据中的图像分别保存到不同的目录
    
    Args:
        file_path: HDF5文件路径
        demo_name: demo名称
        output_base_dir: 基础输出目录
    """
    # 创建三个子目录
    obs_dir = os.path.join(output_base_dir, "obs")
    obs_of_dir = os.path.join(output_base_dir, "obs_of")
    obs_sam_dir = os.path.join(output_base_dir, "obs_sam")
    
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(obs_of_dir, exist_ok=True)
    os.makedirs(obs_sam_dir, exist_ok=True)
    
    with h5py.File(file_path, 'r') as f:
        if demo_name not in f:
            print(f"Demo {demo_name} 不存在于文件中")
            return
        
        demo_group = f[demo_name]
        
        # 读取数据
        obs = demo_group['obs'][:]           # (T, H, W, 3)
        obs_of = demo_group['obs_of'][:]     # (T, H, W, 3) - RGB格式
        obs_sam = demo_group['obs_sam'][:]   # (T, H, W, 3) - RGB格式
        
        T = obs.shape[0]
        print(f"保存 {demo_name}: {T} 帧图像")
        print(f"obs shape: {obs.shape}")
        print(f"obs_of shape: {obs_of.shape}")
        print(f"obs_sam shape: {obs_sam.shape}")
        
        # 获取文件名前缀（从文件路径中提取环境名称）
        env_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        file_idx = os.path.splitext(os.path.basename(file_path))[0]
        prefix = f"{env_name}_{file_idx}_{demo_name}"
        
        # 保存所有帧的图像
        for frame_idx in range(T):
            frame_name = f"{prefix}_frame_{frame_idx:04d}.png"
            
            # 保存原始观测图像
            obs_image = obs[frame_idx]  # (H, W, 3)
            obs_pil = Image.fromarray(obs_image.astype(np.uint8))
            obs_path = os.path.join(obs_dir, frame_name)
            obs_pil.save(obs_path)
            
            # 保存光流RGB图像
            obs_of_image = obs_of[frame_idx]  # (H, W, 3)
            obs_of_pil = Image.fromarray(obs_of_image.astype(np.uint8))
            obs_of_path = os.path.join(obs_of_dir, frame_name)
            obs_of_pil.save(obs_of_path)
            
            # 保存分割后的光流RGB图像
            obs_sam_image = obs_sam[frame_idx]  # (H, W, 3)
            obs_sam_pil = Image.fromarray(obs_sam_image.astype(np.uint8))
            obs_sam_path = os.path.join(obs_sam_dir, frame_name)
            obs_sam_pil.save(obs_sam_path)
            
            if frame_idx % 20 == 0:  # 每20帧打印一次进度
                print(f"  已保存第 {frame_idx+1}/{T} 帧")
        
        print(f"✓ 图像保存完成！")
        print(f"  原始图像: {obs_dir} ({T} 张)")
        print(f"  光流图像: {obs_of_dir} ({T} 张)")
        print(f"  分割光流图像: {obs_sam_dir} ({T} 张)")
        
        return obs_dir, obs_of_dir, obs_sam_dir


def visualize_demo_data(file_path, demo_name, output_dir="./visualization"):
    """
    可视化demo数据，将obs、obs_of、obs_sam在同一行显示并保存为GIF
    
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
        obs = demo_group['obs'][:]           # (T, H, W, 3)
        obs_of = demo_group['obs_of'][:]     # (T, H, W, 3) - 现在是RGB格式
        obs_sam = demo_group['obs_sam'][:]   # (T, H, W, 3) - 现在是RGB格式
        
        T = obs.shape[0]
        print(f"可视化 {demo_name}: {T} 帧")
        print(f"obs shape: {obs.shape}")
        print(f"obs_of shape: {obs_of.shape}")
        print(f"obs_sam shape: {obs_sam.shape}")
        
        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # fig.suptitle(f'{demo_name} Visualization', fontsize=16)
        
        # 初始化显示
        im1 = axes[0].imshow(obs[0], animated=True)
        # axes[0].set_title('Original Observation')
        axes[0].axis('off')
        
        # 光流可视化 - 现在直接显示RGB图像
        im2 = axes[1].imshow(obs_of[0], animated=True)
        # axes[1].set_title('Optical Flow RGB')
        axes[1].axis('off')
        
        # SAM mask光流可视化 - 现在直接显示RGB图像
        im3 = axes[2].imshow(obs_sam[0], animated=True)
        # axes[2].set_title('Masked Optical Flow RGB')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        def animate(frame):
            # 更新obs
            im1.set_array(obs[frame])
            
            # 更新obs_of - 直接使用RGB图像
            im2.set_array(obs_of[frame])
            
            # 更新obs_sam - 直接使用RGB图像
            im3.set_array(obs_sam[frame])
            
            return [im1, im2, im3]
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=T, interval=200, blit=True, repeat=True)
        
        # 保存为GIF
        output_path_gif = os.path.join(output_dir, f"{demo_name}_visualization.gif")
        writer_gif = animation.PillowWriter(fps=10, metadata=dict(artist='Demo Visualization'))
        
        try:
            print(f"正在保存GIF动画...")
            anim.save(output_path_gif, writer=writer_gif)
            print(f"✓ GIF动画已保存: {output_path_gif}")
        except Exception as e:
            print(f"✗ GIF保存失败: {e}")
        
        # 保存为MP4
        output_path_mp4 = os.path.join(output_dir, f"{demo_name}_visualization.mp4")
        writer_mp4 = animation.FFMpegWriter(fps=10, metadata=dict(artist='Demo Visualization'), bitrate=1800)
        
        try:
            print(f"正在保存MP4视频...")
            anim.save(output_path_mp4, writer=writer_mp4)
            print(f"✓ MP4视频已保存: {output_path_mp4}")
        except Exception as e:
            print(f"✗ MP4保存失败: {e}")
        
        plt.close(fig)
        return output_path_gif


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
        count = 0
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

            count += 1
            if count >= 1:
                print("\n仅显示前1个demo的信息...")
                break


if __name__ == "__main__":
    # 指定文件路径和输出目录
    file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/bigfish/train/0.hdf5"
    file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/chaser/train/3.hdf5"
    # file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/dodgeball/train/0.hdf5"
    file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/heist/train/2.hdf5"
    file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/leaper/train/2.hdf5"
    file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow_512/bigfish/train/0.hdf5"
    # file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow_512/chaser/train/0.hdf5"
    file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow_512/leaper/train/0.hdf5"
    # file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow_512/heist/train/0.hdf5"
    # file_path = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow/maze/train/0.hdf5"
    
    output_base_dir = "/root/Docker_RoboLearn/Benchmark/LAPO_IRL/Test/videos"
    demo_name = "demo1"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        exit(1)
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"开始处理文件: {file_path}")
    print(f"输出基础目录: {output_base_dir}")
    
    # 先测试文件结构
    test_hdf5_structure(file_path)
    
    print(f"\n开始保存 {demo_name} 的图像...")
    
    # 保存图像到分类目录
    obs_dir, obs_of_dir, obs_sam_dir = save_demo_images(file_path, demo_name, output_base_dir)
    
    print(f"\n开始可视化 {demo_name}...")
    
    # 可视化指定demo（生成GIF）
    gif_path = visualize_demo_data(file_path, demo_name, output_base_dir)
    
    if gif_path and os.path.exists(gif_path):
        print(f"\n✓ 处理完成！")
        print(f"图像已分别保存到:")
        print(f"  原始图像: {obs_dir}")
        print(f"  光流图像: {obs_of_dir}")
        print(f"  分割光流图像: {obs_sam_dir}")
        print(f"GIF文件保存在: {gif_path}")
    else:
        print(f"\n✗ 处理失败！")
