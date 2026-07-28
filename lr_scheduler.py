import torch
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(optimizer, scheduler_type='cosine_warmup', total_epochs=50,
                    warmup_epochs=5, n_iter_per_epoch=1,
                    min_lr=5e-6, warmup_lr=5e-7):
    """
    Build learning rate scheduler.

    'cosine' and 'cosine_warmup' both use timm's CosineLRScheduler with
    per-step updates (t_in_epochs=False), matching the original training setup.
    Call scheduler.step_update(global_step) after every optimizer step.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'cosine' | 'cosine_warmup' | 'none'
        total_epochs: Total training epochs
        warmup_epochs: Number of warmup epochs
        n_iter_per_epoch: Number of iterations per epoch
        min_lr: Minimum learning rate (lr_min at end of schedule)
        warmup_lr: Initial learning rate during warmup

    Returns:
        timm CosineLRScheduler instance
    """
    if scheduler_type in ('cosine', 'cosine_warmup'):
        num_steps = int(total_epochs * n_iter_per_epoch)
        warmup_steps = int(warmup_epochs * n_iter_per_epoch)
        return CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type!r}. Use 'cosine', 'cosine_warmup', or 'none'.")
