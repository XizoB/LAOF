import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio
import h5py
from omegaconf import OmegaConf
from pathlib import Path
import os
import doy
import sys
import cv2
from PIL import Image
import argparse

import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 添加RAFT路径
sys.path.append('/root/Docker_RoboLearn/Benchmark/LAPO_IRL/RAFT/core')
from raft import RAFT
from raft_utils.utils import InputPadder
from raft_utils import flow_viz


# 添加LangSAM路径
sys.path.append('/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lang-segment-anything')
from lang_sam import LangSAM

from utils.data_loader import normalize_obs
from torch.distributions import Categorical
from utils.utils import create_decoder, set_random_seed
from utils import paths, env_utils, utils


def init_models():
    """
    初始化RAFT光流模型和LangSAM模型
    """
    # 创建RAFT参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/root/Docker_RoboLearn/Benchmark/LAPO_IRL/RAFT/models/raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    # 解析参数（使用默认值）
    raft_args = parser.parse_args([])  # 传入空列表使用默认值
    
    # 初始化RAFT模型
    raft_model = torch.nn.DataParallel(RAFT(raft_args))
    raft_model.load_state_dict(torch.load(raft_args.model))
    raft_model = raft_model.module
    raft_model.to(config.DEVICE)
    raft_model.eval()
    
    # 初始化LangSAM模型
    langsam_model = LangSAM()
    
    return raft_model, langsam_model


def flow_to_rgb(flow, method="opencv", sigma=0.01):
    """
    将光流数据转换为RGB图像
    
    Args:
        flow: numpy array (H, W, 2) - 光流数据
        method: str - 转换方法 ("opencv" 或 "paper")
    
    Returns:
        rgb_image: numpy array (H, W, 3) - RGB图像 (0-255)
    """
    if method == "opencv":
        # 原始OpenCV方法
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 创建HSV图像
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        
        # 色调(Hue)由角度决定
        hsv[..., 0] = angle * 180 / np.pi / 2
        
        # 饱和度(Saturation)设为最大值
        hsv[..., 1] = 255
        
        # 明度(Value)由幅度决定，需要归一化到0-255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # 转换HSV到RGB
        rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    elif method == "paper":
        # 论文方法
        u = flow[..., 0]  # 水平位移
        v = flow[..., 1]  # 垂直位移
        
        # 1. 方向 → 色相 (Hue)
        alpha = np.arctan2(-v, -u)  # [-π, π]
        # alpha = np.arctan2(v, u)  # [-π, π]
        H = (alpha + np.pi) / (2 * np.pi)  # [0, 1]
        
        # 2. 速度大小 → 饱和度/亮度
        m = np.sqrt(u**2 + v**2)  # 幅值
        
        # 归一化参数
        H_img, W_img = flow.shape[:2]
        max_flow = sigma * np.sqrt(H_img**2 + W_img**2)
        
        # 归一化到 [0,1]
        m_norm = np.minimum(1.0, m / max_flow)
        
        # 将归一化的幅值同时作为饱和度和亮度
        S = m_norm
        V = m_norm
        
        # 创建HSV图像 (值域 [0,1])
        # hsv_float = np.stack([H, S, V], axis=-1)
        hsv_float = np.stack([H * 179 / 255, S, V], axis=-1)

        # 转换为 [0,255] 范围的HSV
        hsv = (hsv_float * 255).astype(np.uint8)
        
        # 转换HSV到RGB
        rgb_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    else:
        raise ValueError(f"不支持的方法: {method}。请选择 'opencv' 或 'paper'")
    
    return rgb_image



def process_optical_flow(obs_raw_sequence, raft_model, flow_viz_method="opencv", sigma=0.01):
    """
    处理光流数据，转换为RGB图像，保持512x512分辨率
    
    Args:
        obs_raw_sequence: numpy array (N, 512, 512, 3)
        raft_model: RAFT模型
        flow_viz_method: str - 光流可视化方法 (保留参数但使用flow_viz)
        sigma: float - 保留参数但使用flow_viz
    
    Returns:
        obs_of: numpy array (N, 512, 512, 3) - 光流RGB图像 (512x512)
    """
    if len(obs_raw_sequence) < 2:
        # 如果只有一帧，直接返回零填充的结果
        obs_of = np.zeros((len(obs_raw_sequence), 512, 512, 3), dtype=np.uint8)
        return obs_of
    
    # 转换为torch tensor格式 (N-1, 3, 512, 512)
    images1 = torch.from_numpy(obs_raw_sequence[:-1]).permute(0, 3, 1, 2).float().to(config.DEVICE)
    images2 = torch.from_numpy(obs_raw_sequence[1:]).permute(0, 3, 1, 2).float().to(config.DEVICE)
    
    flow_results = []
    
    with torch.no_grad():
        for i in range(len(images1)):
            image1 = images1[i:i+1]  # (1, 3, 512, 512)
            image2 = images2[i:i+1]  # (1, 3, 512, 512)
            
            padder = InputPadder(image1.shape)
            image1_pad, image2_pad = padder.pad(image1, image2)
            
            # 计算光流
            flow_low, flow_up = raft_model(image1_pad, image2_pad, iters=20, test_mode=True)
            
            # 去除padding
            flow = padder.unpad(flow_up)  # (1, 2, 512, 512)
            flow_results.append(flow)
    
    # 合并所有光流结果 (N-1, 2, 512, 512)
    if flow_results:
        all_flows = torch.cat(flow_results, dim=0)
        
        # 添加最后一帧零填充 (N, 2, 512, 512)
        zero_frame = torch.zeros((1, 2, 512, 512), device=all_flows.device, dtype=all_flows.dtype)
        all_flows_padded = torch.cat([all_flows, zero_frame], dim=0)
        
        # 转换为 (N, 512, 512, 2) 格式用于处理
        flow_hwc = all_flows_padded.permute(0, 2, 3, 1).cpu().numpy()  # (N, 512, 512, 2)
        
        # 将光流转换为RGB图像，保持512x512分辨率
        flow_rgb_512_list = []
        

        if flow_viz_method == "opencv":
            for flow in flow_hwc:
                # 使用flow_viz函数将光流转换为RGB图像
                flow_rgb = flow_viz.flow_to_image(flow)  # (512, 512, 3)
                flow_rgb_512_list.append(flow_rgb)
        else:
            for flow in flow_hwc:
                # 使用自定义的flow_to_rgb函数将光流转换为RGB图像
                flow_rgb = flow_to_rgb(flow, method=flow_viz_method, sigma=sigma)  # (512, 512, 3)
                flow_rgb_512_list.append(flow_rgb)

        obs_of = np.array(flow_rgb_512_list)  # (N, 512, 512, 3)
    else:
        # 如果没有光流结果，返回零填充
        obs_of = np.zeros((len(obs_raw_sequence), 512, 512, 3), dtype=np.uint8)
    
    return obs_of


def process_langsam(obs_raw_sequence, langsam_model, text_prompt="object"):
    """
    处理LangSAM数据，生成mask
    
    Args:
        obs_raw_sequence: numpy array (N, 512, 512, 3)
        langsam_model: LangSAM模型
        text_prompt: 分割文本提示
    
    Returns:
        masks_array: numpy array (N, 1, 512, 512) - 分割mask (布尔值)
    """
    # 将numpy array转换为PIL Image列表
    image_pil_list = []
    for frame in obs_raw_sequence:
        # 确保数据类型为uint8
        frame_uint8 = frame.astype(np.uint8)
        pil_image = Image.fromarray(frame_uint8, mode='RGB')
        image_pil_list.append(pil_image)
    
    # 创建文本提示列表
    text_prompts = [text_prompt] * len(obs_raw_sequence)
    
    # 批量预测
    try:
        results = langsam_model.predict(image_pil_list, text_prompts)
    except:
        # 如果批量预测失败，尝试逐帧预测
        results = []
        for pil_img in image_pil_list:
            try:
                result = langsam_model.predict([pil_img], [text_prompt])
                results.extend(result)
            except:
                # 如果预测失败，添加空结果
                results.append({'masks': np.zeros((1, 512, 512), dtype=bool)})
    
    # 处理mask结果 - 创建 (N, 1, 512, 512) 格式的mask
    all_masks = []
    for result in results:
        if 'masks' in result and len(result['masks']) > 0:
            masks = result['masks']  # 可能是 (K, 512, 512)
            
            if len(masks.shape) == 3 and masks.shape[0] > 1:
                # 多个mask (K, 512, 512) -> 叠加为 (512, 512)
                # 使用逻辑或操作将所有mask叠加在一起
                combined_mask = np.any(masks, axis=0)  # (512, 512)
            elif len(masks.shape) == 3 and masks.shape[0] == 1:
                # 单个mask (1, 512, 512) -> (512, 512)
                combined_mask = masks[0]
            elif len(masks.shape) == 2:
                # 已经是 (512, 512) 格式
                combined_mask = masks
            else:
                # 其他情况，创建空mask
                combined_mask = np.zeros((512, 512), dtype=bool)
            
            # 转换为布尔值并添加通道维度 (1, 512, 512)
            mask_bool = combined_mask.astype(bool)[np.newaxis, ...]
        else:
            # 没有检测到任何对象，创建空mask
            mask_bool = np.zeros((1, 512, 512), dtype=bool)
        
        all_masks.append(mask_bool)
    
    # 转换为numpy array (N, 1, 512, 512)
    masks_array = np.array(all_masks)
    
    # 确保输出格式正确 (N, 1, 512, 512)
    if len(masks_array.shape) != 4 or masks_array.shape[1] != 1 or masks_array.shape[2] != 512 or masks_array.shape[3] != 512:
        print(f"警告: masks_array形状不符合预期，当前形状: {masks_array.shape}")
        # 如果形状不对，重新调整
        if len(masks_array.shape) == 3:  # (N, 512, 512)
            masks_array = masks_array[:, np.newaxis, :, :]  # 添加通道维度
        
    return masks_array


def apply_mask_to_flow(flow_rgb_data, masks_array):
    """
    将mask应用到光流RGB图像上，保持512x512分辨率，背景设为白色
    
    Args:
        flow_rgb_data: numpy array (N, 512, 512, 3) - 光流RGB图像数据
        masks_array: numpy array (N, 1, 512, 512) - 分割mask (布尔值)
    
    Returns:
        obs_sam: numpy array (N, 512, 512, 3) - mask应用到光流RGB图像后的结果，背景为白色 (512x512)
    """
    # 应用mask到光流RGB图像
    masked_flow_rgb_list = []
    
    for i in range(len(flow_rgb_data)):
        # 获取当前帧的光流RGB图像 (512, 512, 3)
        flow_rgb_frame = flow_rgb_data[i]  # (512, 512, 3)
        
        # 获取对应的mask (1, 512, 512)
        if i < len(masks_array):
            mask_frame = masks_array[i, 0]  # (512, 512) - 布尔值
        else:
            mask_frame = np.zeros((512, 512), dtype=bool)
        
        # 将布尔mask转换为uint8并扩展到3个通道
        mask_rgb = np.stack([mask_frame.astype(np.uint8)] * 3, axis=-1)  # (512, 512, 3)
        
        # 将mask应用到RGB图像 (逐元素相乘)
        masked_flow_rgb = flow_rgb_frame * mask_rgb  # (512, 512, 3)
        
        masked_flow_rgb_list.append(masked_flow_rgb)
    
    # 转换为numpy array (N, 512, 512, 3)
    obs_sam = np.array(masked_flow_rgb_list)
    
    return obs_sam



def load_policy(policy_path):
    """
    加载已保存策略，并重构相关模块。
    """
    # 加载保存的字典，包含训练时保存的 state_dict、cfg 等
    checkpoint = torch.load(policy_path, weights_only=False)
    cfg = checkpoint["cfg"]
    
    # 根据模型结构重新创建策略，注意 action_dim 与 latent action 维度保持一致
    policy = utils.create_policy(
        cfg.model,
        action_dim=cfg.model.la_dim,
        strict_loading=True,
    )

    # 重构 decoder 模块
    policy.decoder = create_decoder(
        in_dim=cfg.model.la_dim,
        out_dim=cfg.model.ta_dim,
        hidden_sizes=(192, 128, 64),
    )
    
    # --- 根据训练代码中“hacky”修改重构策略架构 ---
    policy.policy_head_sl = policy.policy_head
    policy.policy_head_rl = nn.Linear(policy.policy_head_sl.in_features, cfg.model.ta_dim).to(config.DEVICE)
    with torch.no_grad():
        policy.policy_head_rl.weight[:] = 0
        policy.policy_head_rl.bias[:] = 0
    
    # print("policy.policy_head_sl:", policy.policy_head_rl.weight[:])

    policy.fc_rl = policy.fc
    policy.fc_sl = nn.Sequential(nn.Linear(policy.fc.in_features, policy.fc.out_features), nn.ReLU()).to(config.DEVICE)
    # policy.fc_sl = nn.Sequential(nn.Linear(policy.fc.in_features, policy.fc.out_features)).to(config.DEVICE)
    # 同步 fc_sl 第一层权重
    policy.fc_sl[0].load_state_dict(policy.fc.state_dict())


    policy.load_state_dict(checkpoint["policy"], strict=True)

    # print("policy.policy_head_sl:", policy.policy_head_rl.weight[:])
    
    # 注意这里在评估时不需要冻结参数，可以保持默认
    policy.eval()  # 设置为 eval 模式
    return policy, cfg



def sample_expert_data(policy, env, raft_model, langsam_model, text_prompt, max_ep_steps=1000, max_ep_returns=10.0, nums_ep=50, nums_dataset=100, save_dir=None, flow_viz_method="opencv", sigma=0.01):
    """
    使用智能体与环境交互进行采样，保存高质量轨迹数据。
    
    Args:
        policy: 训练好的策略模型
        env: 交互环境
        raft_model: RAFT光流模型
        langsam_model: LangSAM模型
        text_prompt: LangSAM文本提示
        max_ep_steps: 单条轨迹与环境交互的最长步骤数
        max_ep_returns: 保存轨迹的回报阈值
        nums_ep: 每个数据集包含的轨迹数量
        nums_dataset: 总共保存的数据集数量
        save_dir: 数据保存目录
        flow_viz_method: 光流可视化方法 ("opencv" 或 "paper")
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 数据收集变量 - 存储所有轨迹
    all_trajectories = []  # 存储所有高质量轨迹
    
    # 当前轨迹数据
    current_traj_obs = []
    current_traj_ta = []
    current_traj_done = []
    current_traj_rewards = []
    current_traj_obs_raw = []
    
    dataset_count = 0
    saved_trajectories = 0
    current_ep_return = 0.0
    current_ep_steps = 0  # 当前轨迹的步数
    
    # 初始化环境
    observe = env.reset()
    done = False
    
    print(f"开始采样专家数据，单轨迹最大步数: {max_ep_steps}, 回报阈值: {max_ep_returns}")
    print(f"计划保存 {nums_dataset} 个数据集，每个数据集包含 {nums_ep} 条轨迹")
    
    while dataset_count < nums_dataset:
        # 转换观测为tensor
        obs_tensor = torch.from_numpy(observe).permute((0, 3, 1, 2)).to(config.DEVICE)
        obs_tensor = normalize_obs(obs_tensor)
        
        with torch.no_grad():
            # 策略推理
            hidden_base = policy.conv_stack(obs_tensor)
            
            # RL 分支
            hidden_rl = F.relu(policy.fc_rl(hidden_base))
            logits_rl = policy.policy_head_rl(hidden_rl)
            
            # SL 分支
            hidden_sl = F.relu(policy.fc_sl(hidden_base))
            logits_sl = policy.decoder(policy.policy_head_sl(hidden_sl))
            
            # 融合两分支
            logits = (logits_rl + logits_sl) / 2
            probs = Categorical(logits=logits)
            action = probs.probs.argmax(dim=-1)
        
        # 获取高分辨率渲染图像
        obs_raw = env.render(mode="rgb_array")
        
        # 存储当前步骤数据 - 现在obs和obs_raw都是512x512分辨率
        current_traj_obs.append(obs_raw)  # (512, 512, 3) - 使用高分辨率图像
        current_traj_ta.append(action.cpu().numpy()[0])  # scalar
        current_traj_obs_raw.append(obs_raw)  # (512, 512, 3) - 保持原有变量名以便后续处理
        
        # 环境交互
        observe, reward, done, info = env.step(action.cpu().numpy())
        
        # 存储奖励和done状态
        current_traj_rewards.append(reward[0])  # scalar
        current_traj_done.append(done[0])  # boolean
        # current_ep_return += reward[0]
        current_ep_return += reward.squeeze()
        
        current_ep_steps += 1
        
        # 轨迹结束条件：done为True 或 达到单轨迹最大步数
        if done[0] or current_ep_steps >= max_ep_steps:
            # 从info中获取正确的回合奖励
            for _, item in enumerate(info):
                if "episode" in item.keys():
                    # 记录回合结束信息
                    print(f"轨迹结束，回报: {item['episode']['r']:.2f}, 步数: {item['episode']['l']}, 原回报: {current_ep_return:.2f}, 原步数: {len(current_traj_obs)}")
                    current_ep_return = item["episode"]["r"]

            # 检查是否满足保存条件：回报阈值 + 最小步数要求
            if current_ep_return >= max_ep_returns and len(current_traj_obs) >= 3:
                doy.print(f"[bold green]保存轨迹，回报: {current_ep_return:.2f}, 步数: {len(current_traj_obs)}, 已保存轨迹数: {saved_trajectories+1}/{nums_ep}[/bold green]")
                
                # 处理obs_raw数据生成obs_of和obs_sam
                obs_raw_array = np.array(current_traj_obs_raw)  # (steps, 512, 512, 3)
                
                print(f"  处理光流数据... obs_raw.shape: {obs_raw_array.shape}")
                obs_of = process_optical_flow(obs_raw_array, raft_model, flow_viz_method=flow_viz_method, sigma=sigma)  # (steps, 512, 512, 3)

                print(f"  处理LangSAM数据... ")
                masks_array = process_langsam(obs_raw_array, langsam_model, text_prompt=text_prompt)  # (steps, 1, 512, 512)
                
                print(f"  应用mask到光流RGB图像... ")
                obs_sam = apply_mask_to_flow(obs_of, masks_array)  # (steps, 512, 512, 3)
                
                # 创建单条轨迹的数据字典
                trajectory_data = {
                    'obs': np.array(current_traj_obs, dtype=np.uint8),      # (steps, 512, 512, 3) - 现在是512分辨率
                    'ta': np.array(current_traj_ta, dtype=np.int64),        # (steps,)
                    'done': np.array(current_traj_done, dtype=bool),        # (steps,)
                    'rewards': np.array(current_traj_rewards, dtype=np.float32),  # (steps,)
                    'ep_returns': np.full(len(current_traj_obs), current_ep_return, dtype=np.float32),  # (steps,)
                    'obs_of': obs_of.astype(np.uint8),                      # (steps, 512, 512, 3) - 光流RGB图像
                    'obs_sam': obs_sam.astype(np.uint8)                     # (steps, 512, 512, 3) - masked光流RGB图像
                }
                
                # 添加到轨迹列表
                all_trajectories.append(trajectory_data)
                saved_trajectories += 1
                
                # 检查是否达到保存数据集的条件
                if saved_trajectories >= nums_ep:
                    save_hdf5_dataset(all_trajectories, save_dir, dataset_count)
                    
                    # 重置数据
                    all_trajectories = []
                    saved_trajectories = 0
                    dataset_count += 1
                    
                    # 检查是否已达到目标数据集数量
                    if dataset_count >= nums_dataset:
                        print(f"已保存 {nums_dataset} 个数据集，采样完成！")
                        break
            else:
                if current_ep_return < max_ep_returns:
                    doy.print(f"轨迹回报不满足阈值，跳过保存 (回报: {current_ep_return:.2f} < {max_ep_returns})")
                elif len(current_traj_obs) < 3:
                    doy.print(f"轨迹步数不足，跳过保存 (步数: {len(current_traj_obs)} < 3)")
                else:
                    doy.print(f"轨迹不满足保存条件，跳过保存")
            
            # 重置轨迹数据
            current_traj_obs, current_traj_ta, current_traj_done, current_traj_rewards, current_traj_obs_raw = [], [], [], [], []
            current_ep_return = 0.0
            current_ep_steps = 0
            
            # 重新开始新轨迹
            if dataset_count < nums_dataset:
                observe = env.reset()
                done = False
    
    # 保存剩余数据（如果有且未达到数据集数量限制）
    if len(all_trajectories) > 0 and dataset_count < nums_dataset:
        save_hdf5_dataset(all_trajectories, save_dir, dataset_count)
        dataset_count += 1
    
    print(f"采样完成！")
    print(f"保存的数据集数量: {dataset_count}")
    print(f"数据集文件: {save_dir}/{dataset_count-1}.hdf5")


def save_hdf5_dataset(trajectories_list, save_dir, dataset_idx):
    """
    保存轨迹数据集到HDF5文件，每个轨迹作为一个demo组
    
    Args:
        trajectories_list: 轨迹列表，每个元素是包含轨迹数据的字典
        save_dir: 保存目录
        dataset_idx: 数据集索引
    """
    save_path = os.path.join(save_dir, f"{dataset_idx}.hdf5")
    
    with h5py.File(save_path, 'w') as f:
        for i, traj_data in enumerate(trajectories_list):
            demo_name = f"demo{i+1}"
            demo_group = f.create_group(demo_name)
            
            # 保存每个轨迹的数据
            demo_group.create_dataset('obs', data=traj_data['obs'], compression='gzip')
            demo_group.create_dataset('ta', data=traj_data['ta'], compression='gzip')
            demo_group.create_dataset('done', data=traj_data['done'], compression='gzip')
            demo_group.create_dataset('rewards', data=traj_data['rewards'], compression='gzip')
            demo_group.create_dataset('ep_returns', data=traj_data['ep_returns'], compression='gzip')
            demo_group.create_dataset('obs_of', data=traj_data['obs_of'], compression='gzip')
            demo_group.create_dataset('obs_sam', data=traj_data['obs_sam'], compression='gzip')
    
    print(f"HDF5数据集已保存: {save_path}")
    print(f"  包含 {len(trajectories_list)} 条轨迹 (demo1 到 demo{len(trajectories_list)})")
    
    # 显示第一条轨迹的数据形状信息
    if trajectories_list:
        first_traj = trajectories_list[0]
        print(f"  示例轨迹数据形状:")
        print(f"    obs.shape: {first_traj['obs'].shape} {type(first_traj['obs'])} (512分辨率观测)")
        print(f"    ta.shape: {first_traj['ta'].shape} {type(first_traj['ta'])}")
        print(f"    done.shape: {first_traj['done'].shape} {type(first_traj['done'])}")
        print(f"    rewards.shape: {first_traj['rewards'].shape} {type(first_traj['rewards'])}")
        print(f"    ep_returns.shape: {first_traj['ep_returns'].shape} {type(first_traj['ep_returns'])}")
        print(f"    obs_of.shape: {first_traj['obs_of'].shape} {type(first_traj['obs_of'])} (光流RGB图像)")
        print(f"    obs_sam.shape: {first_traj['obs_sam'].shape} {type(first_traj['obs_sam'])} (masked光流RGB图像)")


def save_dataset(obs_list, ta_list, done_list, rewards_list, ep_returns_list, obs_raw_list, save_dir, dataset_idx):
    """
    保存数据集到npz文件 (已弃用，保留以防兼容性需要)
    """
    # 转换为numpy数组
    obs_array = np.array(obs_list, dtype=np.uint8)  # (N, 64, 64, 3)
    ta_array = np.array(ta_list, dtype=np.int64)    # (N,)
    done_array = np.array(done_list, dtype=bool)     # (N,)
    rewards_array = np.array(rewards_list, dtype=np.float32)  # (N,)
    ep_returns_array = np.array(ep_returns_list, dtype=np.float32)  # (N,)
    obs_raw_array = np.array(obs_raw_list, dtype=np.uint8)  # (N, 512, 512, 3) -> 转为object数组节省空间
    
    # 保存文件
    save_path = os.path.join(save_dir, f"{dataset_idx}.npz")
    np.savez_compressed(
        save_path,
        obs=obs_array,
        ta=ta_array,
        done=done_array,
        rewards=rewards_array,
        ep_returns=ep_returns_array,
        obs_raw=obs_raw_array
    )
    
    print(f"数据集已保存: {save_path}")
    print(f"  obs.shape: {obs_array.shape} {type(obs_array)}")
    print(f"  ta.shape: {ta_array.shape} {type(ta_array)}")
    print(f"  done.shape: {done_array.shape} {type(done_array)}")
    print(f"  rewards.shape: {rewards_array.shape} {type(rewards_array)}")
    print(f"  ep_returns.shape: {ep_returns_array.shape} {type(ep_returns_array)}")
    print(f"  obs_raw.shape: {obs_raw_array.shape} {type(obs_raw_array)}")






if __name__ == "__main__":
    # 设置随机种子
    seed = 66
    set_random_seed(seed)

    # 设定要加载的策略文件路径
    base_cfg_path = "/root/Docker_RoboLearn/Benchmark/LAPO/lapo/conf/defualt.yaml"
    base_cfg = OmegaConf.load(base_cfg_path)
    policy_path = paths.get_decoded_policy_path(config.get(file_cfg=base_cfg).exp_name)
    print(f"Loading policy from {policy_path}")
    
    # 初始化模型
    print("初始化RAFT和LangSAM模型...")
    raft_model, langsam_model = init_models()
    print("模型初始化完成！")
    
    # 加载训练好的策略和配置信息
    policy, cfg = load_policy(policy_path)
    
    # 构建评估用的环境
    # 这里我们可以调用 env_utils 的构造函数，假定与训练时使用的环境相同
    # 注意：如果训练时环境是 vectorized 的，评估时建议使用单个环境实例（num_envs=1）以便逐回合统计回报
    env = env_utils.setup_procgen_env(
        num_envs=1,
        env_id=cfg.env_name,
        gamma=cfg.stage3.gamma,
        rand_seed=seed,
        render_mode="rgb_array",
    )

    
    # 如果返回的是 VecEnv，取第一个环境实例进行评估（这里假定环境符合 Gym 接口）
    if hasattr(env, "reset"):
        eval_env = env
        print("env是个单独的Env")
    else:
        eval_env = env.envs[0]
        print("env是VecEnv 取用envs[0]")
    
    # 数据采样参数设置
    max_ep_steps = 1000  # 单条轨迹与环境交互的最长步骤数（减小用于测试）
    text_prompt_config = {
        "bigfish": "green.",
        "chaser": "purple.",
        "dodgeball": "purple-white hair.",
        "heist": "orange-white hair.",
        "leaper": "green-frog.",
        "maze": "all.",
    }
    max_ep_returns_config = {
        "bigfish": 40,
        "chaser": 13,
        "dodgeball": 20,
        "heist": 10,
        "leaper": 10,
        "maze": 10,
    }
    sigma_config = {
        "bigfish": 0.01,
        "chaser": 0.01,
        "dodgeball": 0.004,
        "heist": 0.01,
        "leaper": 0.005,
        "maze": 0.01,
    }
    nums_ep_config = {
        # "bigfish": 1,
        # "chaser": 100,
        # "dodgeball": 200,
        # "heist": 600,
        # "leaper": 300,
        # "maze": 700,
        "bigfish": 5,
        "chaser": 5,
        "dodgeball": 5,
        "heist": 20,
        "leaper": 1,
        "maze": 5,
    }
    nums_dataset_config = {
        "bigfish": 1,
        # "bigfish": 11,
        "chaser": 1,
        # "chaser": 10,
        "dodgeball": 1,
        "heist": 1,
        "leaper": 1,
        "maze": 1,
    }


    text_prompt = text_prompt_config[f"{cfg.env_name}"]
    max_ep_returns = max_ep_returns_config[f"{cfg.env_name}"]
    nums_ep = nums_ep_config[f"{cfg.env_name}"]
    nums_dataset = nums_dataset_config[f"{cfg.env_name}"]  # 总共保存的数据集数量（0.hdf5到99.hdf5）
    sigma = sigma_config[f"{cfg.env_name}"]  # 总共保存的数据集数量（0.hdf5到99.hdf5）
    print(f"环境: {cfg.env_name}, LangSAM文本提示: '{text_prompt}', 轨迹回报阈值: {max_ep_returns}, 每数据集轨迹数: {nums_ep}, 数据集数量: {nums_dataset}, 光流归一化sigma: {sigma}")

    # 根据环境名称动态设置保存路径
    save_dir = f"/root/Docker_RoboLearn/Benchmark/LAPO_IRL/lapo/expert_data/opticalflow_512/{cfg.env_name}/train"
    print(f"数据保存目录: {save_dir}")
    
    # 光流可视化方法选择
    flow_viz_method = "opencv"  # 可选 "opencv" 或 "paper"
    flow_viz_method = "paper"  # 可选 "opencv" 或 "paper"
    print(f"使用光流可视化方法: {flow_viz_method}")
    
    # 开始采样专家数据
    sample_expert_data(
        policy=policy,
        env=eval_env, 
        raft_model=raft_model,
        langsam_model=langsam_model,
        text_prompt=text_prompt,
        max_ep_steps=max_ep_steps,
        max_ep_returns=max_ep_returns,
        nums_ep=nums_ep,
        nums_dataset=nums_dataset,
        save_dir=save_dir,
        flow_viz_method=flow_viz_method,
        sigma=sigma
    )
    

    print("专家数据采样完成！")