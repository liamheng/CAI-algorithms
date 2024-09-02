from .base_scheduler import BaseScheduler

class LinearScheduler(BaseScheduler):
    def __init__(self, opt, optimizer):
        super().__init__(opt, optimizer)

    def step(self, n_epochs, n_iters, update_position):

        if update_position == self.opt.lr_update_by:
            if self.opt.lr_update_by == 'iteration' and n_iters % self.opt.lr_update_freq == 0:
                ratio = 1.0 - max(0, n_iters / float(self.opt.total_batches * self.opt.epoch_end))
                for param in self.optimizer.param_groups:
                    param['lr'] = self.opt.lr * ratio
            if self.opt.lr_update_by == 'epoch' and n_epochs % self.opt.lr_update_freq == 0:
                ratio = 1.0 - max(0, n_epochs / float(self.opt.epoch_end))
                for param in self.optimizer.param_groups:
                    param['lr'] = self.opt.lr * ratio
            else:
                return

