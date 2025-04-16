from .base_scheduler import BaseScheduler

class ConstantScheduler(BaseScheduler):
    def __init__(self, opt, optimizer):
        super().__init__(opt, optimizer)

    def step(self, n_epochs, n_iters, update_position):
        return

