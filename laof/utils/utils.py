
import doy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader

from doy import loop

import config
from utils import data_loader
from agent.models import FlowDecoder_WOX1, WorldModelShared, FlowDecoderBidirectional, MotionDecoderWithPos, FlowDecoderRes, DecodeModel, WorldModelRes, FlowDecoder, IDM, IDM_Continue, Policy, Policy_one, WorldModel, WorldModel_Continue, SharedEncoder, Actor, Critic, Actor_Independent, Critic_Independent, FlowModel


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

def obs_to_img(obs: Tensor) -> Tensor:
    return ((obs.permute(1, 2, 0) + 0.5) * 255).to(torch.uint8).numpy(force=True)


def create_decoder(in_dim, out_dim, device=config.DEVICE, hidden_sizes=(128, 128)):
    decoder = []
    in_size = h = in_dim
    for h in hidden_sizes:
        decoder.extend([nn.Linear(in_size, h), nn.ReLU()])
        in_size = h
    decoder.append(nn.Linear(h, out_dim))
    return nn.Sequential(*decoder).to(device)



def create_decoder_one(in_dim, out_dim, device=config.DEVICE, hidden_sizes=(128, 128)):
    decoder = []
    in_size = h = in_dim
    for h in hidden_sizes:
        decoder.extend([nn.Linear(in_size, h), nn.ReLU()])
        in_size = h
    decoder.append(nn.Linear(h, out_dim))
    return nn.Sequential(*decoder).to(device)



def create_dynamics_models(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (2 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_dynamics_models_onehorizon(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm

def create_dynamics_models_bc(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])

    return idm

def create_dynamics_continue_models(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth


    idm = IDM_Continue(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_dynamics_models_onehorizon_wm_res(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModelRes(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth + 3,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_dynamics_models_onehorizon_wm_separate(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_models_onehorizon_wmflow_shared(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModelShared(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_dynamics_models_onehorizon_flow(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)


    fm = FlowDecoder(
        latent_dim=model_cfg.la_dim,
        hidden_channels=64,
        num_iters=5,
        img_hw=64,
        use_coords=True
    ).to(config.DEVICE)


    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_models_flow(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoder_WOX1(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_continue_models_flow(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM_Continue(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoder_WOX1(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_models_singlechannelflow(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoder_WOX1(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=1,  # 单通道光流输出
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_flowautoencoder(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoder_WOX1(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_models_flow_decode(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoder_WOX1(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    ad = create_decoder(
        in_dim=model_cfg.la_dim,
        out_dim=model_cfg.ta_dim,
        hidden_sizes=(192, 128, 64),
    )

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])
        ad.load_state_dict(state_dicts["ad"])

    return idm, wm, fm, ad


def create_dynamics_continue_models_flow_decode(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM_Continue(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoder_WOX1(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    ad = create_decoder(
        in_dim=model_cfg.la_dim,
        out_dim=model_cfg.ta_dim,
        hidden_sizes=(192, 128, 64),
    )

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])
        ad.load_state_dict(state_dicts["ad"])

    return idm, wm, fm, ad


def create_dynamics_models_onehorizon_flow_bidirectional(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoderBidirectional(
        latent_dim=model_cfg.la_dim,
        hidden_channels=64,
        num_iters=5,
        img_hw=64,
        use_coords=True
    ).to(config.DEVICE)


    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_models_onehorizon_flow_res(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    fm = FlowDecoderRes(
        latent_dim=model_cfg.la_dim,
        hidden_channels=64,
        num_iters=5,
        img_hw=64,
        use_coords=True
    ).to(config.DEVICE)


    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        fm.load_state_dict(state_dicts["fm"])

    return idm, wm, fm


def create_dynamics_models_onehorizon_decode_res(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    rm = DecodeModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)


    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        rm.load_state_dict(state_dicts["rm"])

    return idm, wm, rm


def create_dynamics_models_onehorizon_decode_res_grid(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    rm = MotionDecoderWithPos(
        model_cfg.la_dim,
        in_depth=2,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)


    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        rm.load_state_dict(state_dicts["rm"])

    return idm, wm, rm


def create_dynamics_models_onehorizon_action_decode(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    ad = create_decoder(
        in_dim=model_cfg.la_dim,
        out_dim=model_cfg.ta_dim,
        hidden_sizes=(192, 128, 64),
    )

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        ad.load_state_dict(state_dicts["ad"])

    return idm, wm, ad


def create_dynamics_continue_models_action_decode(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM_Continue(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    ad = create_decoder(
        in_dim=model_cfg.la_dim,
        out_dim=model_cfg.ta_dim,
        hidden_sizes=(192, 128, 64),
    )

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        ad.load_state_dict(state_dicts["ad"])

    return idm, wm, ad


def create_dynamics_models_flow_action_decoder(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (2 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    ad = create_decoder(
        in_dim=model_cfg.la_dim,
        out_dim=model_cfg.ta_dim,
        hidden_sizes=(192, 128, 64),
    )

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])
        ad.load_state_dict(state_dicts["ad"])

    return idm, wm, ad


def create_dynamics_models_continue_onehorizon(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM_Continue, WorldModel_Continue]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM_Continue(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel_Continue(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_dynamics_models_one(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (0 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_dynamics_models_two(
    model_cfg: config.ModelConfig, state_dicts: dict | None = None
) -> tuple[IDM, WorldModel]:
    obs_depth = 3
    idm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_in_depth = obs_depth * (1 + config.ADD_TIME_HORIZON)
    wm_out_depth = obs_depth

    idm = IDM(
        model_cfg.vq,
        (idm_in_depth, 64, 64),
        model_cfg.la_dim,
        model_cfg.idm_impala_scale,
    ).to(config.DEVICE)

    wm = WorldModel(
        model_cfg.la_dim,
        in_depth=wm_in_depth,
        out_depth=wm_out_depth,
        base_size=model_cfg.wm_scale,
    ).to(config.DEVICE)

    if state_dicts is not None:
        idm.load_state_dict(state_dicts["idm"])
        wm.load_state_dict(state_dicts["wm"])

    return idm, wm


def create_policy(
    model_cfg: config.ModelConfig,
    action_dim: int,
    policy_in_depth: int = 3,
    state_dict: dict | None = None,
    strict_loading: bool = True,
):
    policy = Policy(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)

    if state_dict is not None:
        policy.load_state_dict(state_dict, strict=strict_loading)

    return policy


def create_policy_one(
    model_cfg: config.ModelConfig,
    action_dim: int,
    policy_in_depth: int = 3,
    state_dict: dict | None = None,
    strict_loading: bool = True,
):
    policy = Policy_one(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
        model_cfg.vq,
    ).to(config.DEVICE)

    if state_dict is not None:
        policy.load_state_dict(state_dict, strict=strict_loading)

    return policy


def create_policy_onehorizon(
    model_cfg: config.ModelConfig,
    action_dim: int,
    policy_in_depth: int = 3,
    state_dict: dict | None = None,
    strict_loading: bool = True,
):
    policy = Policy_one(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
        model_cfg.vq,
    ).to(config.DEVICE)

    if state_dict is not None:
        policy.load_state_dict(state_dict, strict=strict_loading)

    return policy


def create_irl_policy(
    model_cfg: config.ModelConfig,
    action_dim: int,
    policy_in_depth: int = 3,
    state_dict: dict | None = None,
    strict_loading: bool = True,
):
    
    sharedencoder = SharedEncoder(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)
    actor = Actor(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)
    critic = Critic(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)
    critic_target = Critic(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    return sharedencoder, actor, critic, critic_target


def create_irl_independent_policy(
    model_cfg: config.ModelConfig,
    action_dim: int,
    policy_in_depth: int = 3,
    state_dict: dict | None = None,
    strict_loading: bool = True,
):
    
    actor = Actor_Independent(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)
    critic = Critic_Independent(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)
    critic_target = Critic_Independent(
        (policy_in_depth, 64, 64),
        action_dim,
        model_cfg.policy_impala_scale,
    ).to(config.DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    return actor, critic, critic_target



def eval_latent_repr(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)

def eval_latent_repr_onehorizon(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked_onehorizon(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)

def eval_latent_repr_onehorizon_res(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked_onehorizon_res(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)

def eval_latent_repr_one(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked_one(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)

def eval_latent_repr_two(labeled_data: data_loader.DataStager, idm: IDM):
    batch = labeled_data.td_unfolded[:131072]
    actions = idm.label_chunked_two(batch).select("ta", "la").to(config.DEVICE)
    return train_decoder(data=actions)

def train_decoder(
    data: TensorDict,  # tensordict with keys "la", "ta"
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(data["la"].shape[-1], TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)

    train_data, test_data = data[: len(data) // 2], data[len(data) // 2 :]

    dataloader = DataLoader(
        train_data,  # type: ignore
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    step = 0
    for i in range(epochs):
        for batch in dataloader:
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            if step % 10 == 0:
                with torch.no_grad():
                    test_pred_ta = decoder(test_data["la"])
                    test_ta = test_data["ta"][:, -2]

                    logger(
                        step=i,
                        test_loss=F.cross_entropy(test_pred_ta, test_ta),
                        test_acc=(test_pred_ta.argmax(-1) == test_ta).float().mean(),
                    )
            step += 1

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=logger["test_acc"][-1],
        test_loss=logger["test_loss"][-1],
    )

    return decoder, metrics






def eval_latent(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics



def eval_latent_continue(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon(batch, training=False)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon(batch, training=False)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics




def eval_latent_continue_last(
    model_cfg,
    eval_data,
    idm: IDM,
    step: None,
    hidden_sizes=(128, 128),
    epochs=50,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    # decoder = create_decoder(int(model_cfg.la_dim * 0.25), TA_DIM, hidden_sizes=hidden_sizes)
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    # opt = torch.optim.AdamW(decoder.parameters())
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-5)
    logger = doy.Logger(use_wandb=True)


    eval_step = 1
    for i in loop(epochs + 1, desc="[green bold](stage-1) Training Action Decoder with TFRecord"):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon(batch, training=False)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step+eval_step,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            eval_step += 1

        # 在最后一个 epoch 结束后进行评估
        if i % 10 == 0:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon(batch, training=False)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

            logger(
                step+eval_step,
                test_acc=avg_acc,
                test_loss=avg_loss,
            )
            eval_step += 1

    return decoder



def eval_latent_idmflow(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon_flowsam(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon_flowsam(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics



def eval_latent_autoflow(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon_autoflow(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon_autoflow(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics




def eval_latent_res(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon_res(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon_res(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics


def eval_latent_res_continue(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon_res(batch, training=False)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon_res(batch, training=False)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics



def eval_latent_onemask(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)
            # 根据倒数第二列筛选, 一次性剔除不满足条件的数据
            mask_col = batch["mask_nums"][:, -2]
            keep_idx = mask_col == 1  # 布尔索引
            batch = batch[keep_idx]

            with torch.no_grad():
                idm.label_onehorizon(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)
                # 根据倒数第二列筛选, 一次性剔除不满足条件的数据
                mask_col = batch["mask_nums"][:, -2]
                keep_idx = mask_col == 1  # 布尔索引
                batch = batch[keep_idx]
                
                with torch.no_grad():
                    idm.label_onehorizon(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics




def eval_latent_idmres(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon_res(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon_res(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics



def eval_latent_idmflow(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon_flowsam(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon_flowsam(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics



def eval_latent_idmflowsam(
    model_cfg,
    eval_data,
    idm: IDM,
    hidden_sizes=(128, 128),
    epochs=3,
    bs=128,
):
    """
    Evaluate the quality of the learned latent representation:
        -> How much information about true actions do latent actions contain?
    """
    TA_DIM = 15
    decoder = create_decoder(model_cfg.la_dim, TA_DIM, hidden_sizes=hidden_sizes)
    opt = torch.optim.AdamW(decoder.parameters())
    logger = doy.Logger(use_wandb=False)


    step = 0
    for i in range(epochs):
        # print(f"Epoch {i+1}/{epochs}")
        for batch in eval_data:
            batch = batch.to(config.DEVICE)

            with torch.no_grad():
                idm.label_onehorizon_flowsam(batch)
            pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
            ta = batch["ta"][:, -2] # torch.Size([128])
            loss = F.cross_entropy(pred_ta, ta)
            opt.zero_grad()
            loss.backward()
            opt.step()

            logger(
                step=i,
                train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
                train_loss=loss,
            )

            step += 1

        # 在最后一个 epoch 结束后进行评估
        if i == epochs - 1:
            total_loss, total_acc, total_batches = 0.0, 0.0, 0
            for batch in eval_data:
                batch = batch.to(config.DEVICE)

                with torch.no_grad():
                    idm.label_onehorizon_flowsam(batch)
                    test_pred_ta = decoder(batch["la"])
                    test_ta = batch["ta"][:, -2]

                # loss 和 acc
                loss = F.cross_entropy(test_pred_ta, test_ta)
                acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

                # 累积
                total_loss += loss.item()
                total_acc += acc.item()
                total_batches += 1
            
            # 计算平均值
            avg_loss = total_loss / total_batches
            avg_acc = total_acc / total_batches

    metrics = dict(
        train_acc=np.mean(logger["train_acc"][-15:]),
        train_loss=np.mean(logger["train_loss"][-15:]),
        test_acc=avg_acc,
        test_loss=avg_loss,
    )

    return decoder, metrics