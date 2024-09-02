class BaseScheduler():
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, optimizer):
        self.optimizer = optimizer
        self.opt = opt

    def step(self, n_epochs, n_iters, update_position):
        raise NotImplementedError