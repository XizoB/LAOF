import doy
import torch
import hydra
from doy import loop
from omegaconf import DictConfig
import torch.nn.functional as F

import config
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
    beta = 0.0

    # 配置wandb
    # run, logger = config.wandb_init("laof_stage1", config.get_wandb_cfg(cfg), wandb_enabled=False)
    run, logger = config.wandb_init("laof_stage1", config.get_wandb_cfg(cfg), wandb_enabled=True)

    # 创建IDM/WM模型
    idm, wm, fm, ad = utils.create_dynamics_continue_models_flow_decode(cfg.model)

    # 加载TFRecord数据
    doy.print("[bold blue]Loading TFRecord data...")
    ratio = cfg.ratio
    train_data, train_noaction_data, test_data, eval_data = tfrecord_data_loader.load_ratiodata(cfg.env_name, ratio=ratio)

    # 创建无限流训练迭代器和有限流测试迭代器
    train_iter = train_data.get_iter(
        batch_size=cfg.stage1.bs, 
        infinite=True,  # 无限流模式，用于训练
        shuffle_buffer=10000
    )
    train_noaction_iter = train_noaction_data.get_iter(
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
            [wm, idm, fm, ad],
        ),
    )

    def train_step():
        idm.train()
        wm.train()
        ad.train()

        lr_sched.step(step)

        batch = next(train_iter).to(next(idm.parameters()).device)

        kl_loss = idm.label_onehorizon(batch)
        wm_loss = wm.label_continue_onehorizon(batch)
        pred_ta = ad(batch["la"]) # torch.Size([128, 15])
        ta = batch["ta"][:, -2] # torch.Size([128])
        ad_loss = F.cross_entropy(pred_ta, ta)
        loss = ratio * (wm_loss + beta * kl_loss + ad_loss)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters(), *ad.parameters()], 2)
        
        opt.step()

        logger(
            step,
            wm_loss=wm_loss,
            global_step=step * cfg.stage1.bs,
            kl_loss=kl_loss,
            ad_loss=ad_loss,
            grad_norm=grad_norm,
            **lr_sched.get_state(),
        )


    def train_noaction_step():
        idm.train()
        wm.train()
        fm.train()

        lr_sched.step(step)

        batch = next(train_noaction_iter).to(next(idm.parameters()).device)

        kl_loss = idm.label_onehorizon(batch)
        wm_loss = wm.label_continue_onehorizon(batch)
        fm_loss = fm.label_continue_flow(batch)

        loss = (1 - ratio) * (wm_loss + beta * kl_loss + fm_loss)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters(), *fm.parameters()], 2)

        opt.step()

        logger(
            step,
            wm_loss=wm_loss,
            global_step=step * cfg.stage1.bs,
            kl_loss=kl_loss,
            fm_loss=fm_loss,
            grad_norm=grad_norm,
            **lr_sched.get_state(),
        )


    def test_step():
        idm.eval()  # disables idm.vq ema update
        wm.eval()
        ad.eval()

        # evaluate IDM + FDM generalization on (action-free) test data
        batch = next(test_iter).to(next(idm.parameters()).device)

        idm.label_onehorizon(batch)
        wm_loss = wm.label_continue_onehorizon(batch)
        fm_loss = fm.label_continue_flow(batch)

        pred_ta = ad(batch["la"]) # torch.Size([128, 15])
        ta = batch["ta"][:, -2] # torch.Size([128])
        ad_loss = F.cross_entropy(pred_ta, ta)

        # train latent -> true action decoder and evaluate its predictiveness
        _, eval_metrics = utils.eval_latent_continue(cfg.model, eval_data, idm)

        logger(step, wm_loss_test=wm_loss, fm_loss_test=fm_loss, ad_loss_test=ad_loss, global_step=step * cfg.stage1.bs, **eval_metrics)



    # 训练循环
    for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM with TFRecord"):
        if step % 2 == 0:
            train_step()
        else:
            train_noaction_step()

        if step % 500 == 0:
            test_step()

        if step > 0 and (step % 5_000 == 0 or step == cfg.stage1.steps):
            torch.save(
                dict(
                    **doy.get_state_dicts(wm=wm, idm=idm, fm=fm, ad=ad, opt=opt),
                    step=step,
                    cfg=cfg,
                    logger=logger,
                ),
                paths.get_models_path(cfg.exp_name),
            )

if __name__ == "__main__":
    main()