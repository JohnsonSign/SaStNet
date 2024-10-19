# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

import torch
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR
from bisect import bisect_right
import math


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
      Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
      Args:
          optimizer (Optimizer): Wrapped optimizer.
          multiplier: init learning rate = base lr / multiplier
          warmup_epoch: target learning rate is reached at warmup_epoch, gradually
          after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
      """

    def __init__(self, optimizer, multiplier, warmup_epoch, after_scheduler, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """

        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


def get_scheduler(optimizer, n_iter_per_epoch, args):
    if "cosine" in args.lr_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.00001,
            T_max=(args.epochs - args.warmup_epoch) * n_iter_per_epoch)
    elif "step" in args.lr_scheduler:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=args.lr_decay_rate,
            milestones=[(m - args.warmup_epoch) * n_iter_per_epoch for m in args.lr_steps])
    else:
        raise NotImplementedError(f"scheduler {args.lr_scheduler} not supported")


    if args.warmup_epoch != 0 :
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.warmup_multiplier,
            after_scheduler=scheduler,
            warmup_epoch=args.warmup_epoch * n_iter_per_epoch)

    return scheduler


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not milestones == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr *
            warmup_factor *
            self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupStepCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        t_total,
        milestones,
        cycles=0.5,
        CosineT_factor=3,
        min_decayed=0.001,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):

        if not milestones == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.t_total = t_total
        self.cycles = cycles
        self.CosineT_factor = CosineT_factor
        self.min_decayed = min_decayed
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupStepCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        cosine_step_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        else:
            step_bin = bisect_right(self.milestones, self.last_epoch)
            step_factor = self.gamma ** step_bin

            if step_bin == 0:
                now_epoch = self.last_epoch - self.warmup_iters
                total_epoch = self.milestones[step_bin] - self.warmup_iters
            elif step_bin < len(self.milestones):
                now_epoch =  self.last_epoch - self.milestones[step_bin-1]
                total_epoch = self.milestones[step_bin] - self.milestones[step_bin-1]
            elif step_bin == len(self.milestones):
                now_epoch =  self.last_epoch - self.milestones[step_bin-1]
                total_epoch = self.t_total - self.milestones[step_bin-1]  

            progress = now_epoch / (self.CosineT_factor * total_epoch)
            cosine_factor = self.cycles * (1 + math.cos(math.pi * progress))
            cosine_step_factor = cosine_factor * step_factor
        
        all_factor = max((warmup_factor * cosine_step_factor), self.min_decayed)

        return [
            base_lr * all_factor for base_lr in self.base_lrs
        ]