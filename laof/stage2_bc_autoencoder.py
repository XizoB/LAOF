import torch
import torch.nn.functional as F
import hydra
import doy
from pathlib import Path
from omegaconf import DictConfig
from doy import loop

import config
from utils import utils, paths, tfrecord_data_loader


@hydra.main(config_path="conf", config_name="defualt.yaml")
def main(cfg: DictConfig):
    
    # 导入IDM/WM 模型参数
    state_dicts = torch.load(paths.get_models_path(config.get(file_cfg=cfg).exp_name), weights_only=False)

    # 导入配置参数
    cfg = config.get(file_cfg=cfg, base_cfg=state_dicts["cfg"], reload_keys=["stage2", "stage3"])
    cfg.stage_exp_name = doy.random_proquint(1)
    doy.print("[bold green]Running LAPO stage 2 (latent behavior cloning) with config:")
    config.print_cfg(cfg)
    utils.set_random_seed(cfg.seed)


    if state_dicts["step"] != cfg.stage1.steps:
        doy.log(
            f"[bold red]Warning: using IDM/WM from incomplete training run {state_dicts['step']}/{cfg.stage1.steps} steps"
        )


    # 创建IDM/WM模型
    idm, _, _ = utils.create_dynamics_flowautoencoder(cfg.model, state_dicts=state_dicts)
    idm.eval()


    # 创建策略模型
    policy = utils.create_policy(cfg.model, cfg.model.la_dim)
    opt, lr_sched = doy.LRScheduler.make(
        policy=(
            doy.PiecewiseLinearSchedule(
                [0, 1000, cfg.stage2.steps + 1], [0.01 * cfg.stage2.lr, cfg.stage2.lr, 0]
            ),
            [policy],
        ),
    )


    # 加载数据
    train_data, test_data, eval_data = tfrecord_data_loader.load(cfg.env_name)
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



    # 评估IDM模型
    # _, eval_metrics = utils.eval_latent(cfg.model, eval_data, idm)
    # doy.log(f"Decoder metrics sanity check: {eval_metrics}")


    # 初始化wandb
    run, logger = config.wandb_init("laof_stage2", config.get_wandb_cfg(cfg), wandb_enabled=True)
    # run, logger = config.wandb_init("laof_stage2", config.get_wandb_cfg(cfg), wandb_enabled=False)

    
    # 训练策略模型
    for step in loop(
        cfg.stage2.steps + 1, desc="[green bold](stage-2) Training latent policy via BC"
    ):
        lr_sched.step(step)

        policy.train()
        batch = next(train_iter).to(next(idm.parameters()).device)
        idm.label_onehorizon_autoflow(batch)

        preds = policy(batch["obs"][:, -2])  # the -2 selects last the pre-transition ob
        loss = F.mse_loss(preds, batch["la"])

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger(
            step=step,
            loss=loss,
            **lr_sched.get_state(),
        )

        # 测试
        if step % 200 == 0:
            policy.eval()
            test_batch = next(test_iter).to(next(idm.parameters()).device)
            idm.label_onehorizon_autoflow(test_batch)
            test_loss = F.mse_loss(policy(test_batch["obs"][:, -2]), test_batch["la"])
            logger(step=step, test_loss=test_loss)


    # 保存最终模型
    torch.save(
        dict(policy=doy.state_dict_orig(policy), cfg=cfg, logger=logger),
        paths.get_latent_policy_path(cfg.exp_name),
    )


if __name__ == "__main__":
    main()
