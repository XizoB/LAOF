import doy
import torch
import hydra
from doy import loop
from omegaconf import DictConfig

import config
import torch.nn.functional as F

from utils import utils, paths
from utils import tfrecord_data_loader  # 导入新的TFRecord数据加载器

@hydra.main(config_path="conf", config_name="defualt.yaml")
def main(cfg: DictConfig):    
    
    # 导入配置参数
    cfg = config.get(file_cfg=cfg)
    cfg.stage_exp_name = doy.random_proquint(1)
    doy.print("[bold green]Running LAPO stage 1 (IDM/FDM training) with TFRecord data:")
    config.print_cfg(cfg)
    utils.set_random_seed(cfg.seed)
    # beta = 0.01
    beta = 0.0002
    # 配置wandb
    # run, logger = config.wandb_init("laof_stage1", config.get_wandb_cfg(cfg), wandb_enabled=False)
    run, logger = config.wandb_init("laof_stage1_continue", config.get_wandb_cfg(cfg), wandb_enabled=True)

    # 创建IDM/WM模型
    idm, _ = utils.create_dynamics_continue_models(cfg.model)
    decoder = utils.create_decoder(128, 15, hidden_sizes=(128, 128))

    # 加载TFRecord数据
    doy.print("[bold blue]Loading TFRecord data...")
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


    doy.print(f"[bold green]Successfully loaded TFRecord data for {cfg.env_name}")
    
    # 创建优化器
    # optimizer and learning rate scheduler
    opt, lr_sched = doy.LRScheduler.make(
        all=(
            doy.PiecewiseLinearSchedule(
                [0, 50, cfg.stage1.steps + 1],
                [0.1 * cfg.stage1.lr, cfg.stage1.lr, 0.01 * cfg.stage1.lr],
            ),
            [decoder],
        ),
    )

    def train_step():
        idm.train()

        lr_sched.step(step)

        batch = next(train_iter).to(next(idm.parameters()).device)

        idm.label_onehorizon_decode(batch)

        pred_ta = decoder(batch["la"]) # torch.Size([128, 15])
        ta = batch["ta"][:, -2] # torch.Size([128])
        loss = F.cross_entropy(pred_ta, ta)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*decoder.parameters()], 2)
        opt.step()

        logger(
            step,
            global_step=step * cfg.stage1.bs,
            action_loss=loss,
            train_acc=(pred_ta.argmax(-1) == ta).float().mean(),
            grad_norm=grad_norm,
            **lr_sched.get_state(),
        )



    def test_step():
        idm.eval()  # disables idm.vq ema update

        with torch.no_grad():
            # evaluate IDM + FDM generalization on (action-free) test data
            batch = next(test_iter).to(next(idm.parameters()).device)

            idm.label_onehorizon(batch)
            test_pred_ta = decoder(batch["la"])
            test_ta = batch["ta"][:, -2]

            # loss 和 acc
            loss = F.cross_entropy(test_pred_ta, test_ta)
            acc = (test_pred_ta.argmax(-1) == test_ta).float().mean()

        logger(step, action_loss_test=loss, action_acc_test=acc, global_step=step * cfg.stage1.bs)


    # 训练循环
    for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM with TFRecord"):
        train_step()

        if step % 500 == 0:
            test_step()

        if step > 0 and (step % 5_000 == 0 or step == cfg.stage1.steps):
            torch.save(
                dict(
                    **doy.get_state_dicts(idm=idm, decoder=decoder, opt=opt),
                    step=step,
                    cfg=cfg,
                    logger=logger,
                ),
                paths.get_models_path(cfg.exp_name),
            )


if __name__ == "__main__":
    main()