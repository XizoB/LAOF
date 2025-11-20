
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import random
import numpy as np
import imageio
import doy
import matplotlib.pyplot as plt
import os
import argparse

import umap
import holoviews as hv
from holoviews.operation.datashader import rasterize
from PIL import Image
import pandas as pd
import colorcet as cc
import datashader as ds
import datashader.transfer_functions as tf
from datashader.mpl_ext import dsshow
from omegaconf import OmegaConf

from doy import loop
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from torch.distributions import Categorical

from utils.utils import create_decoder, create_policy, create_dynamics_models
from utils import utils, env_utils, tfrecord_data_loader, paths
from utils.data_loader import normalize_obs


hv.extension('bokeh')


def set_random_seed(seed: int, env=None):
    """设置 Python、NumPy 和 PyTorch 的随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 如果你使用 GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 设置 cudnn 为确定性算法（可能会降低部分运行速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_latent(latent_path, defualt_cfg=None):
    state_dict = torch.load(latent_path, weights_only=False)
    cfg = config.get(file_cfg=defualt_cfg, base_cfg=state_dict["cfg"], reload_keys=["stage3"])
    cfg.stage_exp_name += doy.random_proquint(1)
    doy.print("[bold green]Running LAPO stage 3 (latent policy decoding) with config:")
    config.print_cfg(cfg)

    # we only need the IDM for training a decoder online
    # if we just do RL, we don't need it
    models_path = paths.get_models_path(cfg.exp_name)
    idm, _ = utils.create_dynamics_models_onehorizon(cfg.model, state_dicts=torch.load(models_path,  weights_only=False))
    idm.eval()
    return idm, cfg


def load_policy(policy_path):
    """
    加载已保存策略，并重构相关模块。
    """
    # 加载保存的字典，包含训练时保存的 state_dict、cfg 等
    checkpoint = torch.load(policy_path, weights_only=False)
    cfg = checkpoint["cfg"]
    
    # 根据模型结构重新创建策略，注意 action_dim 与 latent action 维度保持一致
    policy = create_policy(
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
    episode_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()    # 假定环境为 Gym 风格，返回初始观测
        done = False
        total_reward = 0.0
        frames = []
        
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
            total_reward += reward.squeeze()
            # print("obs:", obs.shape)
            # print("reward:", reward.squeeze())
            frames.append(obs[0])
            
            for substep, item in enumerate(info):
                if "episode" in item.keys():
                    # 记录回合结束信息
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
        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")
        print("len(frames):", len(frames))
        print("===========================")
        imageio.mimsave(f"exp_results/videos/coinrun_{ep + 1}_reward_{total_reward:.2f}.gif", frames, fps=20)

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return episode_rewards


def evaluate_latent(idm, policy, env, cfg, args):
    train_data, test_data, _ = tfrecord_data_loader.load(cfg.env_name)

    # 创建无限流训练迭代器和有限流测试迭代器
    train_iter = train_data.get_iter(
        batch_size=cfg.stage1.bs, 
        infinite=True,  # 无限流模式，用于训练
        shuffle_buffer=10000
    )
    test_iter = test_data.get_iter(
        batch_size=128, 
        infinite=True,  # 无限流模式，用于测试
        shuffle_buffer=1000
    )

    # _, eval_metrics = utils.eval_latent_repr(train_data, idm)
    # doy.log(f"Decoder metrics sanity check: {eval_metrics}")

    LA = []
    LABLE = []
    LA_ACTION = []
    ACTION_TRUE = []

    # 32768x80/128=20480
    # 4096x16/128=20480
    for step in loop(512, desc="[green bold](stage-2) Training latent policy via BC"
    ):
        # lr_sched.step(step)
        batch = next(test_iter).to(next(idm.parameters()).device)
        idm.label_onehorizon(batch)
        LA.append(batch["la"].detach().cpu().numpy())

        # BC latent policy
        latent_action = latent_select(policy, batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
        LA_ACTION.append(latent_action.detach().cpu().numpy())

        # RL decode policy
        action_lables = action_select(policy, batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
        LABLE.append(action_lables.detach().cpu().numpy())

        # # RL IDM- decode policy
        # action_lables = idm_policy_action_select(policy, batch["la"],)  # the -2 selects last the pre-transition ob
        # LABLE.append(action_lables.detach().cpu().numpy())
        # print("action_lables:", batch["ta"].shape)
        ACTION_TRUE.append(batch["ta"][:, -2].detach().cpu().numpy())



    la = np.concatenate(LA)
    label = np.concatenate(LABLE)
    la_action = np.concatenate(LA_ACTION)
    ture_action = np.concatenate(ACTION_TRUE)
    # label = np.expand_dims(label, axis=1)
    print("la.shape:", la.shape)
    print("label.shape:", label.shape)
    print("la_action.shape:", la_action.shape)
    print("ture_action.shape:", ture_action.shape)

    num_classes = cfg.model.ta_dim


    # 3. UMAP降到2D
    umap_model = umap.UMAP(n_neighbors=80, min_dist=0.1, random_state=42)
    if args == "idm":
        embedding = umap_model.fit_transform(la)
    elif args == "bc":
        embedding = umap_model.fit_transform(la_action)



    # 4. 转DataFrame
    df = pd.DataFrame(embedding, columns=['x', 'y'])
    # df['label'] = label
    # df['label'] = label.astype(str)  # 转换为字符串类型
    df['label'] = ture_action.astype(str)  # 转换为字符串类型



    # 5. Points对象
    points = hv.Points(df, kdims=['x', 'y'], vdims=['label'])



    # 6. 定义颜色映射
    palette = cc.glasbey_light  # 256种明亮颜色
    color_map = {str(i): palette[(i * ((len(palette) - num_classes) // num_classes)) % len(palette)] for i in range(num_classes)}
    print(color_map)
    


    # 7. 用rasterize来像素化，每点一个小格子
    rasterized = rasterize(points, aggregator=ds.count_cat('label'), width=800, height=800)  # 你可以调节width/height控制像素密度
    # 后处理：替换默认颜色
    rasterized = rasterized.opts(
        cmap=color_map,
        colorbar=False
    )
    print('UMAP rasterized with pixel-square points.')



    # 8. 保存
    dir_path = f'exp_results/latent_eval/{cfg.env_name}'
    os.makedirs(dir_path, exist_ok=True)
    split_name = cfg.exp_name.split('/')[1]
    if args == "idm":
        hv.save(rasterized, f"{dir_path}/{split_name}_idm.html", fmt='html', backend='bokeh')
    elif args == "bc":
        hv.save(rasterized, f"{dir_path}/{split_name}_bc.html", fmt='html', backend='bokeh')




def latent_select(policy, obs_tensor):
    with torch.no_grad():
        hidden_base = policy.conv_stack(obs_tensor)

        # SL 分支 forward
        hidden_sl = F.relu(policy.fc_sl(hidden_base))
        # 注意：此处 policy.policy_head_sl 采用的是训练时 freeze 的分支
        logits_sl = policy.policy_head_sl(hidden_sl)

    return logits_sl


def action_select(policy, obs_tensor):
    with torch.no_grad():
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
    return action





if __name__ == "__main__":
    # 添加终端参数支持
    # parser = argparse.ArgumentParser(description="Evaluate trained policy and IDM.")
    # parser.add_argument("--latent", type=str, default="idm", help="Number of evaluation episodes (0 for auto)")
    # args = parser.parse_args()
    args = "idm"
    # args = "bc"

    # 设置随机种子
    seed = 45
    set_random_seed(seed)

    # 设定要加载的策略文件路径
    defualt_cfg_path = "/root/Docker_RoboLearn/Benchmark/LAPO/lapo/conf/defualt.yaml"
    defualt_cfg = OmegaConf.load(defualt_cfg_path)

    # 设定要加载的策略文件路径, 加载 IDM 模型
    policy_path = paths.get_decoded_policy_path(config.get(file_cfg=defualt_cfg).exp_name)
    latent_path = paths.get_latent_policy_path(config.get(file_cfg=defualt_cfg).exp_name)
    print(f"Loading policy from {policy_path}")
    print(f"Loading latent from {latent_path}")

    
    # 加载训练好的策略和配置信息
    policy, _ = load_policy(policy_path)
    idm, cfg = load_latent(latent_path, defualt_cfg=defualt_cfg)

    
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
    # evaluate_policy(policy, eval_env, num_episodes=0)
    evaluate_latent(idm, policy, eval_env, cfg, args)