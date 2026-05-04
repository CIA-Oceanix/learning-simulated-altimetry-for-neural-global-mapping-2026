import torch


def cosanneal_lr_adam(lit_mod, lr, T_max, weight_decay=0.):
    opt = torch.optim.AdamW(
        [{"params": lit_mod.solver.parameters(), "lr": lr},],
        weight_decay=weight_decay,
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max
        ),
    }
