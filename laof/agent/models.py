import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia
import cv2

import config
from utils import data_loader
from funcy import partition
from tensordict import TensorDict
from torch.distributions import Categorical
from torch import distributions as pyd
from typing import Tuple, Dict, Any, List


ObsShapeType = tuple[int, int, int]  # (channels, height, width)


# -------------------- Geometry utils -------------------- #
def coords_grid(batch: int, ht: int, wd: int, device=None):
    y, x = torch.meshgrid(
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=0).float()  # (2,H,W)
    return grid[None].repeat(batch, 1, 1, 1)   # (B,2,H,W)


def normalize_grid(grid: torch.Tensor, H: int, W: int) -> torch.Tensor:
    gx = 2 * (grid[..., 0] / max(W - 1, 1)) - 1
    gy = 2 * (grid[..., 1] / max(H - 1, 1)) - 1
    return torch.stack([gx, gy], dim=-1)


def warp(img: torch.Tensor, flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Backwarp img with flow (positive right/down)."""
    B, C, H, W = img.shape
    base = coords_grid(B, H, W, device=img.device)
    grid = (base + flow).permute(0, 2, 3, 1)   # (B,H,W,2)
    grid = normalize_grid(grid, H, W)
    return F.grid_sample(img, grid, mode=mode, padding_mode="border", align_corners=True)


# -------------------- Census + losses -------------------- #
def census_transform(img: torch.Tensor, patch: int = 7, eps: float = 1e-3) -> torch.Tensor:
    """Dense census transform over patch; channel-wise and concatenated.
    Memory-efficient implementation that processes patches one by one.
    img: (B,C,H,W) in [0,1]
    returns: (B, C*(patch*patch-1), H, W)
    """
    B, C, H, W = img.shape
    r = patch // 2
    img_pad = F.pad(img, (r, r, r, r), mode="reflect")
    
    # Get center pixels for comparison
    center = img  # (B, C, H, W)
    
    # Process each patch position separately to save memory
    census_features = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue  # Skip center pixel
            
            # Extract neighbor patch
            y_start, y_end = r + dy, H + r + dy
            x_start, x_end = r + dx, W + r + dx
            neighbor = img_pad[:, :, y_start:y_end, x_start:x_end]
            
            # Compute normalized difference
            diff = neighbor - center
            normalized_diff = diff / torch.sqrt(diff * diff + eps * eps)
            census_features.append(normalized_diff)
    
    # Concatenate all features along channel dimension
    return torch.cat(census_features, dim=1)


def charbonnier(x: torch.Tensor, eps: float = 1e-3, alpha: float = 0.45) -> torch.Tensor:
    return (x * x + eps * eps) ** alpha


def photometric_census_loss(I1: torch.Tensor, I2: torch.Tensor, flow12: torch.Tensor,
                            valid_mask: torch.Tensor | None = None, patch: int = 7) -> torch.Tensor:
    """Bidirectional photometric loss term for one direction (I1->I2)."""
    I2_w = warp(I2, -flow12)
    
    # Compute census transforms with memory cleanup
    c1 = census_transform(I1, patch)
    c2 = census_transform(torch.clamp(I2_w, 0, 1), patch)
    
    # Clear intermediate tensors to save memory
    del I2_w
    
    # dist = torch.abs(c1 - c2)
    dist = F.mse_loss(c1, c2, reduction='none')
    del c1, c2  # Clean up after use
    
    loss = charbonnier(dist).mean(dim=1, keepdim=True)
    del dist
    
    if valid_mask is not None:
        loss = loss * valid_mask
        denom = valid_mask.sum() + 1e-6
        return loss.sum() / denom
    return loss.mean()
    


def second_order_smoothness(flow: torch.Tensor) -> torch.Tensor:
    def grad2(x):
        dx = x[:, :, :, 2:] - 2 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
        dy = x[:, :, 2:, :] - 2 * x[:, :, 1:-1, :] + x[:, :, :-2, :]
        dxy = (
            x[:, :, 2:, 2:] - x[:, :, 2:, 1:-1] - x[:, :, 1:-1, 2:] + x[:, :, 1:-1, 1:-1]
        )
        return dx, dy, dxy

    u, v = flow[:, 0:1], flow[:, 1:2]
    loss = 0
    for comp in (u, v):
        dx, dy, dxy = grad2(comp)
        loss += charbonnier(dx).mean() + charbonnier(dy).mean() + charbonnier(dxy).mean()
    return loss


def fb_consistency_mask(f12: torch.Tensor, f21: torch.Tensor, alpha1: float = 0.01, alpha2: float = 0.5):
    """Forward-backward consistency based occlusion masks (valid=1-occluded)."""
    f21_w = warp(f21, -f12)
    fb = f12 + f21_w
    mag = f12.pow(2).sum(1, True) + f21_w.pow(2).sum(1, True)
    occ12 = (fb.pow(2).sum(1, True) > alpha1 * mag + alpha2).float()
    valid12 = 1 - occ12

    f12_w = warp(f12, -f21)
    fb2 = f21 + f12_w
    mag2 = f21.pow(2).sum(1, True) + f12_w.pow(2).sum(1, True)
    occ21 = (fb2.pow(2).sum(1, True) > alpha1 * mag2 + alpha2).float()
    valid21 = 1 - occ21
    return valid12, valid21


def merge_TC_dims(x: torch.Tensor):
    """x.shape == (B, T, C, H, W) -> (B, T*C, H, W)"""
    # 使用reshape而不是view来处理不连续的tensor
    return x.reshape(x.shape[0], -1, *x.shape[3:])


class ResidualLayer(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_out_dim, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_out_dim, kernel_size, stride, padding),
        )

    def forward(self, x):
        return x + self.res_block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.conv = nn.Conv2d(in_depth, out_depth, 3, 1, padding=1)
        self.norm = nn.BatchNorm2d(out_depth)
        self.res = ResidualLayer(out_depth, out_depth // 2, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.pool(self.res(self.norm(self.conv(x)))))


class UpsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_depth, out_depth, 2, 2)
        # maybe put bn before conv since that's where unet connections are catted
        self.norm = nn.BatchNorm2d(out_depth)
        self.res = ResidualLayer(out_depth, out_depth // 2, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res(self.norm(self.conv(x))))


class WorldModel(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        # downscaling
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        _, _, h, w = state.shape
        action = action[:, :, None, None] # [128, 128] -> [128, 128, 1, 1]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1) # [128, 131, 64, 64]

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, state], dim=1)) # torch.Size([128, 3, 64, 64])
        return F.tanh(out) / 2

    def label(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, :-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"] = self(wm_in_seq, la) # [128, 3, 64, 64]
        return F.mse_loss(batch["wm_pred"], wm_targ)


    def label_onehorizon(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"] = self(wm_in_seq, la) # [128, 3, 64, 64]
        return F.mse_loss(batch["wm_pred"], wm_targ)

    def label_continue_onehorizon(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"] = self(wm_in_seq, la) # [128, 3, 64, 64]
        return F.mse_loss(batch["wm_pred"], wm_targ)
    

    def label_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        # flow = batch["obs_of"][:, -2] # [128, 3, 64, 64] flow(t->t+1)
        flow = batch["obs_sam"][:, -2] # [128, 3, 64, 64] flow(t->t+1)

        return F.mse_loss(batch["wm_pred_res"], flow)


class DecodeModel(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        # downscaling
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        _, _, h, w = state.shape
        action = action[:, :, None, None] # [128, 128] -> [128, 128, 1, 1]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1) # [128, 131, 64, 64]

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, state], dim=1)) # torch.Size([128, 3, 64, 64])
        return F.tanh(out) / 2


    def label_onehorizon_vq(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return F.mse_loss(batch["wm_pred_res"], batch_obs[:,1])


    def label_onehorizon(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return F.mse_loss(batch["wm_pred_res"], batch_obs[:,1])


class MotionDecoderWithPos(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        # downscaling
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))


        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

        # 预构建标准化坐标网格
        xs = torch.linspace(-1, 1, steps=64, device='cuda')
        ys = torch.linspace(-1, 1, steps=64, device='cuda')
        grid_y, grid_x = torch.meshgrid(ys, xs)
        self.grid_init = torch.stack([grid_x, grid_y], dim=0)  # (2,64,64)
        self.register_buffer("grid_init", self.grid_init)  # 单张量，形状 (2,64,64)


    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        b, _, h, w = state.shape
        action = action[:, :, None, None] # [128, 128] -> [128, 128, 1, 1]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        # x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1) # [128, 131, 64, 64]
        grid_init = self.grid_init.unsqueeze(0).expand(b, -1, -1, -1)  # (B,2,64,64)
        x = torch.cat([grid_init, action.repeat(1, 1, h, w)], dim=1) # [128, 131, 64, 64]

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, grid_init], dim=1)) # torch.Size([128, 3, 64, 64])
        return F.tanh(out) / 2


    def label_onehorizon_vq(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return F.mse_loss(batch["wm_pred_res"], batch_obs[:,1])


    def label_onehorizon(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return F.mse_loss(batch["wm_pred_res"], batch_obs[:,1])


class WorldModelRes(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        # downscaling
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        _, _, h, w = state.shape
        action = action[:, :, None, None] # [128, 128] -> [128, 128, 1, 1]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1) # [128, 131, 64, 64]

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, state], dim=1)) # torch.Size([128, 5, 64, 64])
        return F.tanh(out[:, :3]) / 2, F.tanh(out[:, 3:]) / 2


    def label_onehorizon(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"], batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return F.mse_loss(batch["wm_pred"], wm_targ), F.mse_loss(batch["wm_pred_res"], batch_obs[:,1])


    def label_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"], batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        flow = batch["obs_of"][:, -2] # [128, 3, 64, 64] flow(t->t+1)
        # flow = batch["obs_sam"][:, -2] # [128, 3, 64, 64] flow(t->t+1)

        return F.mse_loss(batch["wm_pred"], wm_targ), F.mse_loss(batch["wm_pred_res"], flow)


class WorldModelShared(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        # downscaling
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

        self.final_conv_flow = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        _, _, h, w = state.shape
        action = action[:, :, None, None] # [128, 128] -> [128, 128, 1, 1]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1) # [128, 131, 64, 64]

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, state], dim=1)) # torch.Size([128, 5, 64, 64])
        out_flow = self.final_conv_flow(torch.cat([x, state], dim=1)) # torch.Size([128, 5, 64, 64])
        return F.tanh(out) / 2, F.tanh(out_flow) / 2

    def label_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"], batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        flow = batch["obs_sam"][:, -2] # [128, 3, 64, 64] flow(t->t+1)

        return F.mse_loss(batch["wm_pred"], wm_targ), F.mse_loss(batch["wm_pred_res"], flow)
    


class FlowDecoder_WOX1(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        # downscaling
        down_sizes = (action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1], b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )


    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        _, _, h, w = state.shape
        action = action[:, :, None, None] # [128, 128] -> [128, 128, 1, 1]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([action.repeat(1, 1, h, w)], dim=1) # [128, 131, 64, 64]

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x], dim=1)) # torch.Size([128, 3, 64, 64])
        return F.tanh(out) / 2

    def label_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        flow = batch["obs_sam"][:, -2] # [128, 3, 64, 64] flow(t->t+1)

        return F.mse_loss(batch["wm_pred_res"], flow)


    def label_continue_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        la = batch["la"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        flow = batch["obs_sam"][:, -2] # [128, 3, 64, 64] flow(t->t+1)

        return F.mse_loss(batch["wm_pred_res"], flow)
    

    def label_singlechannelflow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred_res"] = self(wm_in_seq, la) # [128, 3, 64, 64]

        flow = batch["obs_sam"][:, -2] # [128, 3, 64, 64] flow(t->t+1)

        return F.mse_loss(batch["wm_pred_res"], flow)


class FlowModel(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size
        self.ssim_loss_fn = kornia.losses.SSIMLoss(window_size=5)

        # downscaling: 注意此处起始通道数是 action_dim，而不是 in_depth + action_dim
        # down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        down_sizes = (out_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        # down_sizes = (action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        # final conv: 不再拼接 state，因此去掉 in_depth
        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + out_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )


        # 构造坐标通道
        xs = torch.linspace(-1, 1, steps=64, device='cuda')
        ys = torch.linspace(-1, 1, steps=64, device='cuda')
        grid_y, grid_x = torch.meshgrid(ys, xs)  # shape (64,64)
        self.grid_init = torch.stack([grid_x, grid_y], dim=0)  # (2,64,64)
        self.grid_init = self.grid_init.unsqueeze(0).expand(128, -1, -1, -1)  # (B,2,64,64)


    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        b , _, h, w = state.shape
        action = action[:, :, None, None] # [B, L] -> [B, L, 1, 1] [128, 128, 1, 1]
        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([self.grid_init, action.repeat(1, 1, h, w)], dim=1) # [128, 130, 64, 64]
        # x = action.repeat(1, 1, h, w) # [128, 128, 64, 64]
        
        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action # 将 action 注入最深层

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        # out = self.final_conv(torch.cat([x, state], dim=1))
        out = self.final_conv(torch.cat([x, self.grid_init], dim=1))
        # out = self.final_conv(x) # torch.Size([128, 2, 64, 64])
        MAX_FLOW = 10.0
        out = torch.clamp(out, -MAX_FLOW, MAX_FLOW)
        return out
    

        
    def warp(self, x1, flow):
        """
        Warp an image or feature map x1 according to the optical flow.

        Args:
            x1:   [B, C, H, W] - source image or feature map
            flow: [B, 2, H, W] - optical flow (in pixels)

        Returns:
            Warped image or feature map [B, C, H, W]
        """
        x1 = merge_TC_dims(x1)
        B, C, H, W = x1.size()
        
        # Create mesh grid
        y, x = torch.meshgrid(
            torch.arange(0, H, device=x1.device),
            torch.arange(0, W, device=x1.device),
            indexing="ij"
        )
        grid = torch.stack((x, y), dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # Apply flow to grid
        vgrid = grid + flow  # [B, 2, H, W]

        # Normalize grid to [-1, 1] for grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Sample using bilinear interpolation
        x1_warped = F.grid_sample(x1, vgrid, align_corners=True, padding_mode='border')

        return x1_warped
    

    def smoothness_loss(self, flow):
        dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        return dx.mean() + dy.mean()


    def label_onehorizon_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["flow_pred"] = self(wm_in_seq, la) # [128, 3, 64, 64]
        batch["fm_pred"] = self.warp(wm_in_seq, batch["flow_pred"])  # warp the last frame using predicted flow


        # loss_photometric = torch.abs(batch["fm_pred"] - wm_targ).mean()
        # loss_ssim = 1 - self.ssim_loss_fn(batch["fm_pred"] + 0.5, wm_targ + 0.5).mean()
        # loss_smooth = self.smoothness_loss(batch["flow_pred"])
        # return  0.85 * loss_ssim + 0.15 * loss_photometric + 0.01 * loss_smooth

        return F.mse_loss(batch["fm_pred"], wm_targ)




class FlowDecoderIRR(nn.Module):
    def __init__(self, latent_dim=128, hidden_channels=64, num_iters=5, img_hw=64, use_coords=True):
        super().__init__()
        self.num_iters = num_iters
        self.img_hw = img_hw
        self.use_coords = use_coords
        self.ssim_loss_fn = kornia.losses.SSIMLoss(window_size=5)

        
        in_channels = latent_dim + 2 + 2 if use_coords else latent_dim + 2  # latent + flow + coords?
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 2, kernel_size=3, padding=1)
        )
        
        # 预构建标准化坐标网格
        xs = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        ys = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid_init = torch.stack([grid_x, grid_y], dim=0)  # (2,64,64)
        self.register_buffer("grid_init", grid_init)  # 单张量，形状 (2,64,64)



    def forward(self, z):
        B = z.size(0) # [128, 128]
        # Spatial broadcast
        z_spatial = z.view(B, -1, 1, 1).expand(-1, -1, self.img_hw, self.img_hw) # [128, 128] -> [128, 128, 64, 64]
        
        # 初始化 flow 为 0
        flow = torch.zeros(B, 2, self.img_hw, self.img_hw, device=z.device, dtype=z.dtype) # [128, 2, 64, 64]
        coords = self.grid_init.unsqueeze(0).expand(128, -1, -1, -1) if self.use_coords else None # [128, 2, 64, 64]
        
        for _ in range(self.num_iters):
            # 拼接 latent, prior flow, 以及可选坐标通道
            inputs = [z_spatial, flow]
            if coords is not None:
                inputs.append(coords)
            x = torch.cat(inputs, dim=1)
            delta = self.res_conv(x)
            flow = flow + delta
        
        return flow # [128, 2, 64, 64]


    def warp(self, x1, flow):
        """
        Warp an image or feature map x1 according to the optical flow.

        Args:
            x1:   [B, C, H, W] - source image or feature map
            flow: [B, 2, H, W] - optical flow (in pixels)

        Returns:
            Warped image or feature map [B, C, H, W]
        """
        x1 = merge_TC_dims(x1)
        B, C, H, W = x1.size()
        
        # Create mesh grid
        y, x = torch.meshgrid(
            torch.arange(0, H, device=x1.device),
            torch.arange(0, W, device=x1.device),
            indexing="ij"
        )
        grid = torch.stack((x, y), dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 2, H, W]

        # Apply flow to grid
        vgrid = grid + flow  # [B, 2, H, W]

        # Normalize grid to [-1, 1] for grid_sample
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # [B, H, W, 2]

        # Sample using bilinear interpolation
        x1_warped = F.grid_sample(x1, vgrid, align_corners=True, padding_mode='border')

        return x1_warped
    

    def smoothness_loss(self, flow):
        dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
        return dx.mean() + dy.mean()


    def label_onehorizon_flow_vq(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 1, 3, 64, 64] Obs(t-1)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["flow_pred"] = self(la) # [128, 2, 64, 64] - 光流只有x,y两个通道
        batch["fm_pred"] = self.warp(wm_in_seq, batch["flow_pred"])  # warp the previous frame using predicted flow


        # loss_photometric = torch.abs(batch["fm_pred"] - wm_targ).mean()
        # loss_ssim = 1 - self.ssim_loss_fn(batch["fm_pred"] + 0.5, wm_targ + 0.5).mean()
        # loss_smooth = self.smoothness_loss(batch["flow_pred"])
        # return  F.mse_loss(batch["fm_pred"], wm_targ) + 0.1 * loss_smooth

        # return loss_photometric
        return F.mse_loss(batch["fm_pred"], wm_targ)


    def label_onehorizon_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 1, 3, 64, 64] Obs(t-1)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t)
        la = batch["la"]  # TODO: also allow using la(noq) [128, 128]
        batch["flow_pred"] = self(la) # [128, 2, 64, 64] - 光流只有x,y两个通道
        batch["fm_pred"] = self.warp(wm_in_seq, batch["flow_pred"])  # warp the previous frame using predicted flow

        return F.mse_loss(batch["fm_pred"], wm_targ)




class FlowDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_channels=64, num_iters=5, img_hw=64, use_coords=True):
        super().__init__()
        self.num_iters = num_iters
        self.img_hw = img_hw
        self.use_coords = use_coords
        self.ssim_loss_fn = kornia.losses.SSIMLoss(window_size=5)

        
        in_channels = latent_dim + 2 + 2 if use_coords else latent_dim + 2  # latent + flow + coords?
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 2, kernel_size=3, padding=1)
        )
        
        # 预构建标准化坐标网格
        xs = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        ys = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid_init = torch.stack([grid_x, grid_y], dim=0)  # (2,64,64)
        self.register_buffer("grid_init", grid_init)  # 单张量，形状 (2,64,64)



    def forward(self, z):
        B = z.size(0) # [128, 128]
        # Spatial broadcast
        z_spatial = z.view(B, -1, 1, 1).expand(-1, -1, self.img_hw, self.img_hw) # [128, 128] -> [128, 128, 64, 64]
        
        # 初始化 flow 为 0
        flow = torch.zeros(B, 2, self.img_hw, self.img_hw, device=z.device, dtype=z.dtype) # [128, 2, 64, 64]
        coords = self.grid_init.unsqueeze(0).expand(128, -1, -1, -1) if self.use_coords else None # [128, 2, 64, 64]
        
        for _ in range(self.num_iters):
            # 拼接 latent, prior flow, 以及可选坐标通道
            inputs = [z_spatial, flow]
            if coords is not None:
                inputs.append(coords)
            x = torch.cat(inputs, dim=1)
            delta = self.res_conv(x)
            flow = flow + delta
        
        return flow # [128, 2, 64, 64]


    def label_onehorizon_flow_vq(self, batch: TensorDict) -> torch.Tensor:
        # wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 1, 3, 64, 64] Obs(t-1)
        wm_targ = batch["obs_sam"][:, -2] * 8.0# [128, 3, 64, 64] Obs(t)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["flow_pred"] = self(la) # [128, 2, 64, 64] - 光流只有x,y两个通道

        return F.mse_loss(batch["flow_pred"], wm_targ)




# -------------------- UnFlow loss aggregator -------------------- #
class UnFlowLoss(nn.Module):
    def __init__(self, census_patch: int = 5, w_photo: float = 1.0, w_smooth: float = 0.1,
                 pyr_weights: Tuple[float, ...] = (0.32, 0.08, 0.02, 0.01)):
        super().__init__()
        self.census_patch = census_patch  # Reduced from 7 to 5 for memory efficiency
        self.w_photo = w_photo
        self.w_smooth = w_smooth
        self.pyr_weights = pyr_weights

    def forward(self, I1s: List[torch.Tensor], I2s: List[torch.Tensor],
                flows12: List[torch.Tensor], flows21: List[torch.Tensor]):
        # Compute occlusion masks on full-res predictions and downsample
        valid12, valid21 = fb_consistency_mask(flows12[0].detach(), flows21[0].detach())
        valids12 = [F.interpolate(valid12, size=f.shape[-2:], mode="nearest") for f in flows12]
        valids21 = [F.interpolate(valid21, size=f.shape[-2:], mode="nearest") for f in flows21]

        loss_ph = 0
        loss_sm = 0
        for w, I1, I2, f12, f21, m12, m21 in zip(self.pyr_weights, I1s, I2s, flows12, flows21, valids12, valids21):
            # Process losses with memory cleanup
            ph_loss_12 = photometric_census_loss(I1, I2, f12, valid_mask=m12, patch=self.census_patch)
            ph_loss_21 = photometric_census_loss(I2, I1, f21, valid_mask=m21, patch=self.census_patch)
            loss_ph += w * (ph_loss_12 + ph_loss_21)
            
            # sm_loss = second_order_smoothness(f12) + second_order_smoothness(f21)
            # loss_sm += w * sm_loss
            
            # Clear intermediate variables
            # del ph_loss_12, ph_loss_21, sm_loss
            del ph_loss_12, ph_loss_21

        # total = self.w_photo * loss_ph + self.w_smooth * loss_sm
        total = self.w_photo * loss_ph
        # return total, {"photo": loss_ph.detach(), "smooth": loss_sm.detach()}
        return total, {"photo": loss_ph.detach()}
        


class FlowDecoderBidirectional(nn.Module):
    def __init__(self, latent_dim=128, 
                 hidden_channels=64, 
                 num_iters=5, 
                 img_hw=64, 
                 use_coords=True,
                 census_patch: int = 7,
                 pyr_scales: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
                 pyr_weights: Tuple[float, ...] = (0.32, 0.08, 0.02, 0.01),
                 w_photo: float = 1.0, w_smooth: float = 0.1):
        super().__init__()
        self.num_iters = num_iters
        self.img_hw = img_hw
        self.use_coords = use_coords
        self.pyr_scales = pyr_scales
        self.criterion = UnFlowLoss(census_patch=census_patch, w_photo=w_photo,
                                    w_smooth=w_smooth, pyr_weights=pyr_weights)
        
        
        in_channels = latent_dim + 2 + 2 if use_coords else latent_dim + 2  # latent + flow + coords?
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 2, kernel_size=3, padding=1)
        )
        
        # 预构建标准化坐标网格
        xs = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        ys = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid_init = torch.stack([grid_x, grid_y], dim=0)  # (2,64,64)
        self.register_buffer("grid_init", grid_init)  # 单张量，形状 (2,64,64)


    def forward(self, z):
        B = z.size(0) # [128, 128]
        # Spatial broadcast
        z_spatial = z.view(B, -1, 1, 1).expand(-1, -1, self.img_hw, self.img_hw) # [128, 128] -> [128, 128, 64, 64]
        
        # 初始化 flow 为 0
        flow = torch.zeros(B, 2, self.img_hw, self.img_hw, device=z.device, dtype=z.dtype) # [128, 2, 64, 64]
        coords = self.grid_init.unsqueeze(0).expand(128, -1, -1, -1) if self.use_coords else None # [128, 2, 64, 64]
        
        for _ in range(self.num_iters):
            # 拼接 latent, prior flow, 以及可选坐标通道
            inputs = [z_spatial, flow]
            if coords is not None:
                inputs.append(coords)
            x = torch.cat(inputs, dim=1)
            delta = self.res_conv(x)
            flow = flow + delta
        
        return flow # [128, 2, 64, 64]

    @staticmethod
    def build_pyramid(img: torch.Tensor, scales: Tuple[float, ...]) -> List[torch.Tensor]:
        outs = []
        for s in scales:
            if s == 1.0:
                outs.append(img)
            else:
                outs.append(F.interpolate(img, scale_factor=s, mode="bilinear", align_corners=True))
        return outs
    

    def label_onehorizon_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 1, 3, 64, 64] Obs(t-1)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        la_reverse = batch["la_q_reverse"]  # TODO: also allow using la(noq) [128, 128]

        batch["flow_pred"] = self(la) # [128, 2, 64, 64] - 光流只有x,y两个通道
        batch["flow_pred_reverse"] = self(la_reverse) # [128, 2, 64, 64] - 光流只有x,y两个通道


        # Build pyramids (from high -> low resolution)
        I1s = self.build_pyramid(batch["obs"][:, -2], self.pyr_scales)
        I2s = self.build_pyramid(batch["obs"][:, -1], self.pyr_scales)

        # Resize flows to each pyramid level (start from full-res)
        flows12 = [F.interpolate(batch["flow_pred"], size=I.shape[-2:], mode="bilinear", align_corners=True) for I in I1s]
        flows21 = [F.interpolate(batch["flow_pred_reverse"], size=I.shape[-2:], mode="bilinear", align_corners=True) for I in I2s]

        loss, logs = self.criterion(I1s, I2s, flows12, flows21)
        return flows12, flows21, loss, logs






class FlowDecoderRes(nn.Module):
    def __init__(self, latent_dim=128, hidden_channels=64, num_iters=5, img_hw=64, use_coords=True):
        super().__init__()
        self.num_iters = num_iters
        self.img_hw = img_hw
        self.use_coords = use_coords
        self.ssim_loss_fn = kornia.losses.SSIMLoss(window_size=5)

        
        in_channels = latent_dim + 3 + 2 if use_coords else latent_dim + 2  # latent + flow + coords?
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, 3, kernel_size=3, padding=1)
        )
        
        # 预构建标准化坐标网格
        xs = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        ys = torch.linspace(-1, 1, steps=img_hw, device='cuda')
        grid_y, grid_x = torch.meshgrid(ys, xs)
        self.grid_init = torch.stack([grid_x, grid_y], dim=0)  # (2,64,64)
        self.grid_init = self.grid_init.unsqueeze(0).expand(128, -1, -1, -1)  # (B,2,64,64)


    def forward(self, z):
        B = z.size(0) # [128, 128]
        # Spatial broadcast
        z_spatial = z.view(B, -1, 1, 1).expand(-1, -1, self.img_hw, self.img_hw) # [128, 128] -> [128, 128, 64, 64]
        
        # 初始化 flow 为 0
        flow = torch.zeros(B, 3, self.img_hw, self.img_hw, device=z.device, dtype=z.dtype) # [128, 3, 64, 64]
        coords = self.grid_init if self.use_coords else None # [128, 2, 64, 64]
        
        for _ in range(self.num_iters):
            # 拼接 latent, prior flow, 以及可选坐标通道
            inputs = [z_spatial, flow]
            if coords is not None:
                inputs.append(coords)
            x = torch.cat(inputs, dim=1)
            delta = self.res_conv(x)
            flow = flow + delta
        
        return F.tanh(flow) / 2 # [128, 3, 64, 64]


    def label_onehorizon_flow(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 1, 3, 64, 64] Obs(t-1)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["flow_pred"] = self(la) # [128, 3, 64, 64] - 光流只有x,y两个通道

        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return F.mse_loss(batch["flow_pred"], batch_obs[:,1])



class WorldModel_Continue(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        # downscaling
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b, 32 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = (32 * b, 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),
        )

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq) # [128, 1, 3, 64, 64] -> [128, 3, 64, 64]

        _, _, h, w = state.shape
        action = action[:, :, None, None]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1)

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, state], dim=1))
        return F.tanh(out) / 2

    def label(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, :-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"] = self(wm_in_seq, la) # [128, 3, 64, 64]
        return F.mse_loss(batch["wm_pred"], wm_targ)


    def label_onehorizon(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, -2:-1] # [128, 3, 3, 64, 64] -> [128, 2, 3, 64, 64] Obs(t-k,...,t-1,t)
        wm_targ = batch["obs"][:, -1] # [128, 3, 64, 64] Obs(t+1)
        la = batch["la_q"]  # TODO: also allow using la(noq) [128, 128]
        batch["wm_pred"] = self(wm_in_seq, la) # [128, 3, 64, 64]
        return F.mse_loss(batch["wm_pred"], wm_targ)






def layer_init(layer, std=None, bias_const=0.0):
    if std is not None:
        std = np.sqrt(2)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# based on https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ImpalaResidualBlock(self._out_channels)
        self.res_block1 = ImpalaResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self) -> tuple[int, int, int]:
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


def get_impala(
    shape: ObsShapeType,
    impala_cnn_scale: int,
    out_channels: tuple[int, ...],
    out_features: int,
) -> tuple[nn.Sequential, nn.Linear]:
    conv_stack = []
    for out_ch in out_channels:
        conv_seq = ConvSequence(shape, impala_cnn_scale * out_ch)
        shape = conv_seq.get_output_shape()
        conv_stack.append(conv_seq)
    conv_stack = nn.Sequential(*conv_stack, nn.Flatten(), nn.ReLU())
    fc = nn.Linear(in_features=np.prod(shape), out_features=out_features)
    return conv_stack, fc


def get_fc(
    shape: ObsShapeType,
    impala_cnn_scale: int,
    out_channels: tuple[int, ...],
    out_features: int,
    action_dim: int,
) -> tuple[nn.Sequential, nn.Linear]:
    conv_stack = []
    for out_ch in out_channels:
        conv_seq = ConvSequence(shape, impala_cnn_scale * out_ch)
        shape = conv_seq.get_output_shape()
        conv_stack.append(conv_seq)

    fc = nn.Linear(in_features=np.prod(shape)+action_dim, out_features=out_features)
    return fc



class Policy(nn.Module):
    """IMPALA CNN-based policy"""

    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        self.conv_stack, self.fc = get_impala(
            obs_shape, impala_scale, impala_channels, impala_features
        )
        self.policy_head = nn.Linear(impala_features, action_dim)
        self.value_head = layer_init(nn.Linear(impala_features, 1), std=1)

    def forward(self, x):
        return self.policy_head(F.relu(self.fc(self.conv_stack(x))))

    def get_value(self, x):
        return self.value_head(F.relu(self.fc(self.conv_stack(x))))

    def get_action_and_value(self, x, action=None):
        hidden = F.relu(self.fc(self.conv_stack(x)))
        probs = Categorical(logits=self.policy_head(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(hidden)



class Policy_one(nn.Module):
    """IMPALA CNN-based policy"""

    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        vq_config: config.VQConfig,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        self.conv_stack, self.fc = get_impala(
            obs_shape, impala_scale, impala_channels, impala_features
        )
        self.policy_head = nn.Linear(impala_features, action_dim)
        self.vq = VQEmbeddingEMA(vq_config)
        # self.value_head = layer_init(nn.Linear(impala_features, 1), std=1)

    def forward(self, x):
        return self.policy_head(F.relu(self.fc(self.conv_stack(x))))

    def get_value(self, x):
        return self.value_head(F.relu(self.fc(self.conv_stack(x))))

    def get_action_and_value(self, x, action=None):
        hidden = F.relu(self.fc(self.conv_stack(x)))
        probs = Categorical(logits=self.policy_head(hidden))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.value_head(hidden)















class SharedEncoder(nn.Module):
    """IMPALA CNN-based policy"""
    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        self.conv_stack, _ = get_impala(
            obs_shape, impala_scale, impala_channels, impala_features
        )

    def forward(self, x):
        return self.conv_stack(x)
    

class Actor(nn.Module):
    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        _, self.fc = get_impala(obs_shape, impala_scale, impala_channels, impala_features)
        self.policy_head = nn.Linear(impala_features, action_dim * 2)


        init_temp = 0.001
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -action_dim
        self.log_std_bounds = [-5, 2]


    def forward(self, x):
        mu, log_std = self.policy_head(F.relu(self.fc(x))).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # self.outputs['mu'] = mu
        # self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, dist.mean



class Critic(nn.Module):
    """IMPALA CNN-based policy"""

    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        self.fc = get_fc(obs_shape, impala_scale, impala_channels, impala_features, action_dim)
        self.value_head = layer_init(nn.Linear(impala_features, 1), std=1)


    def forward(self, x, action):
        x = torch.cat([x, action], dim=-1)
        q = self.value_head(F.relu(self.fc(x)))
        return q
    


class Actor_Independent(nn.Module):
    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        self.encode = SharedEncoder(obs_shape, action_dim, impala_scale, impala_channels, impala_features)
        _, self.fc = get_impala(obs_shape, impala_scale, impala_channels, impala_features)
        self.policy_head = nn.Linear(impala_features, action_dim * 2)


        init_temp = 0.001
        self.log_alpha = torch.tensor(np.log(init_temp))
        self.alpha = self.log_alpha.exp()
        self.target_entropy = -action_dim
        self.log_std_bounds = [-5, 2]


    def forward(self, x):
        x = self.encode(x)
        mu, log_std = self.policy_head(F.relu(self.fc(x))).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        # self.outputs['mu'] = mu
        # self.outputs['std'] = std

        dist = SquashedNormal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_prob, dist.mean



class Critic_Independent(nn.Module):
    """IMPALA CNN-based policy"""

    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()
        self.encode = SharedEncoder(obs_shape, action_dim, impala_scale, impala_channels, impala_features)
        self.fc = get_fc(obs_shape, impala_scale, impala_channels, impala_features, action_dim)
        self.value_head = layer_init(nn.Linear(impala_features, 1), std=1)
        # self.value_head = nn.Linear(impala_features, 1)


    def forward(self, x, action):
        x = self.encode(x)
        x = torch.cat([x, action], dim=-1)
        q = self.value_head(F.relu(self.fc(x)))
        return q


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu











class VQEmbeddingEMA(nn.Module):
    def __init__(
        self,
        cfg: config.VQConfig,
        epsilon=1e-5,
    ):
        """
        EMA 的核心思想：不靠反向传播更新 codebook，而是用滑动平均更新
        通过统计“每个 code 被选中的次数”和“被选中时 x 的平均值”
        然后用这些统计量更新 embedding 向量（而不是通过 loss 反传）
        这 本质上就是用数据驱动的方式更新 codebook 向量，而不是用 loss 去“拉”它们。

        码本结构： codebook的数量为2, embedding的数量为64, 每个embedding的维度为16
        ################ ################
        ################ ################
        ################ ################
        ################ ################
        ################ ################

        """
        super(VQEmbeddingEMA, self).__init__()
        self.epsilon = epsilon
        self.cfg = cfg

        # 这里的 embedding 是一个二维的矩阵，第一维是 codebook 的数量，embedding的数量为64, 每个embedding的维度为16
        embedding = torch.zeros(cfg.num_codebooks, cfg.num_embs, cfg.emb_dim) # [2, 64, 16]   ((64)^2)^4
        embedding.uniform_(-1 / cfg.num_embs * 5, 1 / cfg.num_embs * 5)

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(cfg.num_codebooks, cfg.num_embs))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward_2d(self, x):
        B, C, H, W = x.size() # torch.Size([128, 32, 4, 1])
        N, M, D = self.embedding.size() # torch.Size([2, 64, 16])
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2) # ([2, 128, 4, 1, 16])
        x_flat = x.detach().reshape(N, -1, D) # torch.Size([2, 512, 16])

        distances = torch.baddbmm(
            torch.sum(self.embedding**2, dim=2).unsqueeze(1) # torch.Size([2, 64, 16])
            + torch.sum(x_flat**2, dim=2, keepdim=True), # # torch.Size([2, 512, 16])
            x_flat,
            self.embedding.transpose(1, 2),
            alpha=-2.0,
            beta=1.0,
        ) # torch.Size([2, 512, 64])
        indices = torch.argmin(distances, dim=-1) # torch.Size([2, 512])

        encodings = F.one_hot(indices, M).float() # torch.Size([2, 512, 64])
        quantized = torch.gather(
            self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )
        quantized = quantized.view_as(x) # torch.Size([2, 128, 4, 1, 16])

        if self.training:
            self.ema_count = self.cfg.decay * self.ema_count + (
                1 - self.cfg.decay
            ) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (
                (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            )
            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = (
                self.cfg.decay * self.ema_weight + (1 - self.cfg.decay) * dw
            )

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.cfg.commitment_cost * e_latent_loss

        quantized = quantized.detach() + (x - x.detach()) # 传递x的梯度

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)
        ) #  perplexity 作为指标，可以监控 codebook 是否陷入 collapse（只用少数几个码字）。

        return (
            quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W),
            loss,
            perplexity.sum(),
            indices.view(N, B, H, W).permute(1, 0, 2, 3),
        )

    def forward(self, x):
        bs = len(x) # torch.Size([128, 128])
        x = x.view(
            bs, # 128
            self.cfg.num_codebooks * self.cfg.emb_dim, # 2 * 16
            self.cfg.num_discrete_latents, # 4
            1,
        ) # torch.Size([128, 128]) -> torch.Size([128, 32, 4, 1])

        z_q, loss, perplexity, indices = self.forward_2d(x) # [128, 32, 4, 1] -> 

        return (
            z_q.view(
                bs,
                self.cfg.num_codebooks
                * self.cfg.num_discrete_latents
                * self.cfg.emb_dim,
            ),
            loss,
            perplexity,
            indices,
        )

    def inds_to_z_q(self, indices):
        """look up quantization inds in embedding"""
        assert not self.training
        N, M, D = self.embedding.size()
        B, N_, H, W = indices.shape
        assert N == N_

        # N ... num_codebooks
        # M ... num_embs
        # D ... emb_dim
        # H ... num_discrete_latents (kinda)

        inds_flat = indices.permute(1, 0, 2, 3).reshape(N, B * H * W)
        quantized = torch.gather(
            self.embedding, 1, inds_flat.unsqueeze(-1).expand(-1, -1, D)
        )
        return (
            quantized.view(N, B, H, W, D).permute(1, 0, 4, 2, 3).reshape(B, N * D, H, W)
        )  # shape is (B, num_codebooks * emb_dim, num_discrete_latents, 1)




class IDM(nn.Module):
    """Quantized inverse dynamics model"""

    def __init__(
        self,
        vq_config: config.VQConfig,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()

        # initialize impala CNN
        self.conv_stack, self.fc = get_impala(
            obs_shape, impala_scale, impala_channels, impala_features
        )
        self.policy_head = nn.Linear(impala_features, action_dim)

        # initialize quantizer
        self.vq = VQEmbeddingEMA(vq_config)

    def forward(self, x):
        """
        x.shape = (B, T, C, H, W)
        the IDM predicts the action between the last and second to last frames (T dim). 预测最后一帧和倒数第二帧之间的动作。
        """
        x = merge_TC_dims(x) # [128, 3, 3, 64, 64] -> [128, 9, 64, 64] Obs(t-k,...,t-1,t,t+1)
        la = self.policy_head(F.relu(self.fc(self.conv_stack(x)))) # [128, 9, 64, 64] -> [128, 128] [batch_size, action_dim]
        la_q, vq_loss, vq_perp, la_qinds = self.vq(la) # [128, 128], 1, 1, torch.Size([128, 2, 4, 1])

        action_dict = TensorDict(
            dict(
                la=la,
                la_q=la_q,
                la_qinds=la_qinds,
            ),
            batch_size=len(la),
        )

        return action_dict, vq_loss, vq_perp

    def forward_reverse(self, x):
        """
        x.shape = (B, T, C, H, W)
        the IDM predicts the action between the last and second to last frames (T dim). 预测最后一帧和倒数第二帧之间的动作。
        """
        x = merge_TC_dims(x) # [128, 3, 3, 64, 64] -> [128, 9, 64, 64] Obs(t-k,...,t-1,t,t+1)
        la_reverse = self.policy_head(F.relu(self.fc(self.conv_stack(x)))) # [128, 9, 64, 64] -> [128, 128] [batch_size, action_dim]
        la_q_reverse, vq_loss, vq_perp, la_qinds_reverse = self.vq(la_reverse) # [128, 128], 1, 1, torch.Size([128, 2, 4, 1])

        action_dict = TensorDict(
            dict(
                la_reverse=la_reverse,
                la_q_reverse=la_q_reverse,
                la_qinds_reverse=la_qinds_reverse,
            ),
            batch_size=len(la_reverse),
        )

        return action_dict, vq_loss, vq_perp


    def label(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, vq_perp = self(batch["obs"])
        batch.update(action_td)
        return vq_loss, vq_perp
    
    def label_onehorizon(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, vq_perp = self(batch["obs"][:, -2:])
        batch.update(action_td)
        return vq_loss, vq_perp

    def label_onehorizon_flow(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        batch_combined = torch.cat([batch["obs"][:, -2:], batch["obs_of"][:, -2:-1]], dim=1)
        action_td, vq_loss, vq_perp = self(batch_combined)
        batch.update(action_td)
        return vq_loss, vq_perp

    def label_onehorizon_autoflow(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, vq_perp = self(batch["obs_sam"][:, -2:-1])
        batch.update(action_td)
        return vq_loss, vq_perp
    
    def label_onehorizon_flowsam(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        batch_combined = torch.cat([batch["obs"][:, -2:-1], batch["obs_sam"][:, -2:-1]], dim=1)
        action_td, vq_loss, vq_perp = self(batch_combined)
        batch.update(action_td)
        return vq_loss, vq_perp
    
    
    def label_onehorizon_reverse(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, vq_perp = self.forward_reverse(batch["obs"][:, [-1, -2]])
        batch.update(action_td)
        return vq_loss, vq_perp
    
    def label_onehorizon_res(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        action_td, vq_loss, vq_perp = self(batch_obs)
        batch.update(action_td)
        return vq_loss, vq_perp
    
    def label_res(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        batch_obs = batch[:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return self(batch_obs)
    
    def label_onehorizon_regularization(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, vq_perp = self(batch["obs"][:, -2:])

        # 反向对称性损失 Reverse Dynamics Symmetry Loss
        obs_swapped = batch["obs"][:, [0, 2, 1], :, :, :]  # shape 保持不变
        action_td_inverse, _, _ = self(obs_swapped[:, -2:])
        reverse_symmetry_loss = (action_td["la"] + action_td_inverse["la"]).pow(2).mean()  # 这里的操作是将正向和反向的动作相加

        # C-PCFC Cosine similarity between the two Past-to-Current and the Future-to-Current loss
        # action_td_pc, _, _ = self(batch["obs"][:, :-1])
        # action_td_fc, _, _ = self(obs_swapped[:, -2:])
        # spcfc_loss = F.cosine_similarity(action_td_pc["la"], action_td_fc["la"], dim=1).mean()
        
        batch.update(action_td)
        # print("Reverse Symmetry Loss:", reverse_symmetry_loss.item())
        # print("SPCFC Loss:", spcfc_loss.item())
        # return vq_loss, vq_perp, reverse_symmetry_loss, spcfc_loss
        return vq_loss, vq_perp, reverse_symmetry_loss
    
    def label_one(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, vq_perp = self(batch["obs"][:, -2:-1])
        batch.update(action_td)
        return vq_loss, vq_perp

    def label_two(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, vq_loss, vq_perp = self(batch["obs"][:, :-1])
        batch.update(action_td)
        return vq_loss, vq_perp
    

    @torch.no_grad()
    def label_chunked(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:

        def _label(batch: TensorDict):
            return self(data_loader.normalize_obs(batch["obs"].to(config.DEVICE)))[0].to(
                batch.device
            )

        action_dicts = torch.cat(list(map(_label, data.split(chunksize))))
        assert len(action_dicts) == len(data)
        data.update(action_dicts)
        return data


    @torch.no_grad()
    def label_chunked_onehorizon(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:

        def _label(batch: TensorDict):
            return self(data_loader.normalize_obs(batch["obs"][:, -2:].to(config.DEVICE)))[0].to(
                batch.device
            )

        action_dicts = torch.cat(list(map(_label, data.split(chunksize))))
        assert len(action_dicts) == len(data)
        data.update(action_dicts)
        return data


    @torch.no_grad()
    def label_chunked_onehorizon_res(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:

        def _label(batch: TensorDict):
            batch_obs_eval = data_loader.normalize_obs(batch["obs"][:, -2:].to(config.DEVICE)).clone()
            batch_obs_eval[:,1] = (batch_obs_eval[:,1] - batch_obs_eval[:,0])/2.0  # 计算最后一帧和倒数第二帧的差值
            return self(batch_obs_eval)[0].to(
                batch.device
            )

        action_dicts = torch.cat(list(map(_label, data.split(chunksize))))
        assert len(action_dicts) == len(data)
        data.update(action_dicts)
        return data


    @torch.no_grad()
    def label_chunked_one(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:

        def _label(batch: TensorDict):
            return self(data_loader.normalize_obs(batch["obs"][:, -2:-1].to(config.DEVICE)))[0].to(
                batch.device
            )

        action_dicts = torch.cat(list(map(_label, data.split(chunksize))))
        assert len(action_dicts) == len(data)
        data.update(action_dicts)
        return data
    

    @torch.no_grad()
    def label_chunked_two(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:

        def _label(batch: TensorDict):
            return self(data_loader.normalize_obs(batch["obs"][:, :-1].to(config.DEVICE)))[0].to(
                batch.device
            )

        action_dicts = torch.cat(list(map(_label, data.split(chunksize))))
        assert len(action_dicts) == len(data)
        data.update(action_dicts)
        return data
    




class IDM_Continue(nn.Module):
    """Quantized inverse dynamics model"""

    def __init__(
        self,
        vq_config: config.VQConfig,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()

        # initialize impala CNN
        self.conv_stack, self.fc = get_impala(
            obs_shape, impala_scale, impala_channels, impala_features
        )

        # one head → [mu, log_var]
        self.policy_head = nn.Linear(impala_features, 2 * action_dim)

        
    def forward(self, x, training=True):
        """
        x.shape = (B, T, C, H, W)
        the IDM predicts the action between the last and second to last frames (T dim). 预测最后一帧和倒数第二帧之间的动作。
        """
        x = merge_TC_dims(x) # [128, 3, 3, 64, 64] -> [128, 9, 64, 64] Obs(t-k,...,t-1,t,t+1)
        la = self.policy_head(F.relu(self.fc(self.conv_stack(x)))) # [128, 9, 64, 64] -> [128, 128] [batch_size, action_dim]

        z_mu, z_var = torch.chunk(la, 2, dim=1)

        # reparameterization
        if training:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        else:
            z_rep = z_mu

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - z_var.exp(), dim=1).mean()

        action_dict = TensorDict(
            dict(
                la=z_rep,
            ),
            batch_size=len(z_rep),
        )

        return action_dict, kl_loss


    def forward_decode(self, x, training=True):
        """
        x.shape = (B, T, C, H, W)
        the IDM predicts the action between the last and second to last frames (T dim). 预测最后一帧和倒数第二帧之间的动作。
        """
        x = merge_TC_dims(x) # [128, 3, 3, 64, 64] -> [128, 9, 64, 64] Obs(t-k,...,t-1,t,t+1)
        z_rep = self.latent_head(F.relu(self.fc(self.conv_stack(x)))) # [128, 9, 64, 64] -> [128, 128] [batch_size, action_dim]

        action_dict = TensorDict(
            dict(
                la=z_rep,
            ),
            batch_size=len(z_rep),
        )
        return action_dict


    
    def label_onehorizon(self, batch: TensorDict, training=True) -> tuple[torch.Tensor, torch.Tensor]:
        action_td, kl_loss = self(batch["obs"][:, -2:], training=training)
        batch.update(action_td)
        return kl_loss
    
    def label_onehorizon_res(self, batch: TensorDict, training=True) -> tuple[torch.Tensor, torch.Tensor]:
        batch_obs = batch["obs"][:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        action_td, kl_loss = self(batch_obs, training=training)

        batch.update(action_td)
        return kl_loss
    
    def label_res(self, batch: TensorDict, training=True) -> tuple[torch.Tensor, torch.Tensor]:
        batch_obs = batch[:, -2:].clone()  # [128, 2, 3, 64, 64]
        batch_obs[:,1] = (batch_obs[:,1] - batch_obs[:,0])/2.0  # [128, 3, 64, 64] 计算最后一帧和倒数第二帧的差值
        return self(batch_obs, training=training)


if __name__ == "__main__":
    import torchinfo

    cfg = config.get(use_cli_args=False, override_args=["env.name=bossfight"])
    obs_shape = (3, 64, 64)
    policy = Policy(obs_shape, 15, 4)
    wm = WorldModel(cfg.model.la_dim, 3, 3, base_size=24)
    idm = IDM(cfg.model.vq, obs_shape, cfg.model.la_dim, 4)

    bs = 10
    obs = torch.randn(bs, 3, 64, 64)
    la = torch.randn(bs, cfg.model.la_dim)

    print("[WM]")
    print(torchinfo.summary(wm, input_data=(obs, la), depth=2))

    print("\n\n[Policy]")
    print(torchinfo.summary(policy, input_data=(obs,), depth=3))

    # torchinfo doesn't work with tensordict outputs
    orig_fwd = idm.forward
    idm.forward = lambda x: orig_fwd(x)[1]
    print("\n\n[IDM]")
    print(torchinfo.summary(idm, input_data=torch.cat([obs, obs]), depth=3))