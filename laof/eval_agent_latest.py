
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import imageio
from omegaconf import OmegaConf
from pathlib import Path
import os

from utils.data_loader import normalize_obs
from torch.distributions import Categorical
from utils.utils import create_decoder, set_random_seed
from utils import paths, env_utils, utils



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



def evaluate_policy(policy, env, num_episodes=10):
    """
    在给定环境 env 中评估策略 policy 的表现，
    执行 num_episodes 个回合，打印每个回合和平均回报。
    """
    REWARDS = []
    REWARDS_RAW = []
    REWARDS_NORM = []
    LEN = []

    for ep in range(num_episodes):
        obs = env.reset()    # 假定环境为 Gym 风格，返回初始观测
        steps = 0 
        done = False
        total_reward = 0.0
        total_reward_raw = 0.0
        frames = []
        frames_high = []
        
        # 评估回合
        print("done:", done)
        print("obs:", obs.shape)
        # for _ in range(1000):
        while not done:
            # 转换为 tensor，添加 batch 维度，并发送到指定设备
            # obs_tensor = torch.tensor(obs, dtype=torch.float32, device=config.DEVICE).permute((0, 3, 1, 2))
            obs_tensor = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(config.DEVICE)
            obs_tensor = normalize_obs(obs_tensor)
            # print("================")
            # print(obs_tensor)
            # print(obs_tensor.shape)

            with torch.no_grad():
                # 得到共享特征表示
                # print("obs_tensor:", obs_tensor.shape)
                hidden_base = policy.conv_stack(obs_tensor)

                # RL 分支 forward
                hidden_rl = F.relu(policy.fc_rl(hidden_base))
                logits_rl = policy.policy_head_rl(hidden_rl)

                # SL 分支 forward
                hidden_sl = F.relu(policy.fc_sl(hidden_base))
                # 注意：此处 policy.policy_head_sl 采用的是训练时 freeze 的分支
                logits_sl = policy.decoder(policy.policy_head_sl(hidden_sl))

                # 融合两分支：平均后得到最终 logits
                logits = (logits_rl + logits_sl) / 2
                probs = Categorical(logits=logits)

                # ✅ 选择最优动作（推荐用于部署）
                action = probs.probs.argmax(dim=-1)

                # # 如果想保留探索行为（不推荐部署时使用）
                # action = probs.sample()

                # print(action)
                # print(probs.probs)
                # print(logits)
                # print("===============")


            # 环境 step 操作，并更新回报
            obs, reward, done, info = env.step(action.cpu().numpy())
            total_reward_raw += reward.squeeze()
            steps += 1
            # print("obs:", obs.shape)
            # print("reward:", reward.squeeze())
            frames.append(obs[0])
            frames_high.append(env.render(mode="rgb_array"))  # (512, 512, 3)

            
            for substep, item in enumerate(info):
                if "episode" in item.keys():
                    # 记录回合结束信息
                    print("total_reward_raw:", total_reward_raw)
                    print("normalize_return:", env.normalize_return(item["episode"]["r"]))
                    print("item[episode][r]:", item["episode"]["r"])
                    print("item[episode][l]:", item["episode"]["l"])
                    print("substep:", substep)
                    print("done:", done)
                    print("len(frames):", len(frames))
                    print("---------------------")
                    break

            # print(info)
            # print("==========================")
            # plt.imshow(obs[0])
            # plt.axis('off')
            # plt.pause(0.01)
        # total_reward = total_reward[0]
        total_reward = item["episode"]["r"]
        normalize_return = env.normalize_return(total_reward)
        REWARDS.append(total_reward)
        REWARDS_RAW.append(total_reward_raw)
        REWARDS_NORM.append(normalize_return)
        LEN.append(len(frames))
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")
        print("len(frames):", len(frames))
        print("===========================")

        print(cfg.exp_name)
        split_name = cfg.exp_name.split('/')[1]
        dir_path = f"exp_results/video_eval/{cfg.env_name}/{split_name}"
        os.makedirs(dir_path, exist_ok=True)
        file_path = Path(f"{dir_path}/ep_{ep + 1}_len_{len(frames)}_reward_{total_reward:.2f}.gif")
        file_high_path = Path(f"{dir_path}/high_ep_{ep + 1}_len_{len(frames_high)}_reward_{total_reward:.2f}.gif")

        # imageio.mimsave(file_path, frames, fps=20)
        # imageio.mimsave(file_high_path, frames_high, fps=20)


    # 统计信息
    avg_reward = sum(REWARDS) / len(REWARDS)
    max_reward = max(REWARDS)
    min_reward = min(REWARDS)
    max_reward_count = REWARDS.count(max_reward)

    raw_avg_reward = sum(REWARDS_RAW) / len(REWARDS_RAW)
    raw_max_reward = max(REWARDS_RAW)
    raw_min_reward = min(REWARDS_RAW)
    raw_max_reward_count = REWARDS_RAW.count(raw_max_reward)

    norm_avg_reward = sum(REWARDS_NORM) / len(REWARDS_NORM)
    norm_max_reward = max(REWARDS_NORM)
    norm_min_reward = min(REWARDS_NORM)
    norm_max_reward_count = REWARDS_NORM.count(norm_max_reward)

    avg_len = np.mean(LEN)
    max_len = max(LEN)
    min_len = min(LEN)
    max_len_count = LEN.count(max_len)

    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
        # 写入内容
    with open(f"{dir_path}/eps_{num_episodes}_len_{avg_len:.2f}_reward_{avg_reward:.2f}.txt", "w", encoding="utf-8") as f:
        f.write("IDM Evaluation Summary:\n")
        f.write(f"Number of Episodes: {num_episodes}\n\n")

        f.write(f"Reward Summary:\n")
        f.write(f"  Avg Reward: {avg_reward:.2f}\n")
        f.write(f"  Max Reward: {max_reward:.2f}\n")
        f.write(f"  Min Reward: {min_reward:.2f}\n")
        f.write(f"  Max Reward Count: {max_reward_count}\n\n")

        f.write(f"Raw Reward Summary:\n")
        f.write(f"  Avg Reward: {raw_avg_reward:.2f}\n")
        f.write(f"  Max Reward: {raw_max_reward:.2f}\n")
        f.write(f"  Min Reward: {raw_min_reward:.2f}\n")
        f.write(f"  Max Reward Count: {raw_max_reward_count}\n\n")

        f.write(f"Normalized Reward Summary:\n")
        f.write(f"  Avg Reward: {norm_avg_reward:.2f}\n")
        f.write(f"  Max Reward: {norm_max_reward:.2f}\n")
        f.write(f"  Min Reward: {norm_min_reward:.2f}\n")
        f.write(f"  Max Reward Count: {norm_max_reward_count}\n\n")

        f.write(f"Episode Length Summary:\n")
        f.write(f"  Avg Length: {avg_len:.2f}\n")
        f.write(f"  Max Length: {max_len:.2f}\n")
        f.write(f"  Min Length: {min_len:.2f}\n")
        f.write(f"  Max Length Count: {max_len_count}\n\n")



if __name__ == "__main__":
    # 设置随机种子
    seed = 88
    set_random_seed(seed)

    # 设定要加载的策略文件路径
    base_cfg_path = "/root/Docker_RoboLearn/Benchmark/LAPO/lapo/conf/defualt.yaml"
    base_cfg = OmegaConf.load(base_cfg_path)
    policy_path = paths.get_decoded_policy_path(config.get(file_cfg=base_cfg).exp_name)
    print(f"Loading policy from {policy_path}")
    
    
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
    )

    
    # 如果返回的是 VecEnv，取第一个环境实例进行评估（这里假定环境符合 Gym 接口）
    if hasattr(env, "reset"):
        eval_env = env
        print("env是个单独的Env")
    else:
        eval_env = env.envs[0]
        print("env是VecEnv 取用envs[0]")
    
    # 评估策略的表现
    evaluate_policy(policy, eval_env, num_episodes=100)