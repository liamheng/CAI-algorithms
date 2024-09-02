import visdom
import os.path as osp
import sys
import logging
from subprocess import Popen, PIPE


# TODO (Done) Code class Visualizer
# TODO (Done) Test Visualizer

class Visualizer():

    def __init__(self, opt):

        # initialize attributes
        base_id = hash(opt.name) % 1000
        self.opt = opt
        self.port = opt.display_port
        self.server = opt.display_server
        self.name = opt.name
        self.losses = {}
        self.loss_window_base_id = base_id + 1
        self.visuals_mapping = {}
        self.visual_window_base_id = base_id
        self.env = self.opt.display_env
        
        # create connection to visdom server
        self.vis = visdom.Visdom(
            server=self.server, 
            port=self.port, 
            env=self.env,
        )
        self.logger = logging.getLogger(self.opt.name)
        if not self.vis.check_connection():
            self.create_visdom_connections()
    
    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """

        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        self.logger.info('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        self.logger.info('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def visualize(self, epoch, n_iters_per_epoch, n_iters, losses, images):
        self.display_images(epoch, images)
        self.display_losses(epoch, n_iters_per_epoch, losses)
        self.print_losses(epoch, n_iters, losses) 

    def display_images(self, epoch, images):
        for i, (name, image) in enumerate(images.items()):
            
            # map visuals with window id
            if not self.visuals_mapping.get(name):
                self.visuals_mapping[name] = self.visual_window_base_id + 2*i
            # display visals
            if "heatmap" in name:
                self.vis.heatmap(
                    X=image,
                    win=self.visuals_mapping[name],
                    opts=dict(
                        caption="%s visual of %s at epoch %d" % (name, self.name, epoch),
                        ),
                    env=self.env
                )
            else:
                self.vis.image(
                    img=image,
                    win=self.visuals_mapping[name],
                    opts=dict(
                        jpgquality=90,
                        caption="%s visual of %s at epoch %d" % (name, self.name, epoch),
                        ),
                    env=self.env
                )

    def display_losses(self, epoch, n_iters_per_epoch, losses):

        epoch_ratio = float(n_iters_per_epoch) / self.opt.total_batches
        for i, (name, value) in enumerate(losses.items()):
            
            # initialize losses data buffer
            if self.losses.get(name) is None:
                self.losses[name] = {'id': self.loss_window_base_id + 2*i, 'data': {'X':[], 'Y':[]}}
            
            # udpate losses
            self.losses[name]['data']['X'].append(epoch + epoch_ratio)
            self.losses[name]['data']['Y'].append(value)

            # display losses
            self.vis.line(
                X=self.losses[name]['data']['X'],
                Y=self.losses[name]['data']['Y'],
                win=self.losses[name]['id'],
                opts={
                    'title': '%s %s loss over time' % (self.name, name),
                    'xlabel': 'epoch',
                    'ylabel': 'loss'
                }
            )

    def print_losses(self, epoch, iters, losses):
        """Print current losses on the terminal"""

        message = "(epoch: %d, iters: %d), losses: " % (epoch, iters)
        for name, value in losses.items():
            message += "%s:\t%.3f\t" % (name, value)
        self.logger.info(message)

class AttrDict(dict):
    """Only for modlue testing of visualizer"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == '__main__':
    import torch
    opt = dict(
        display_port=8080,
        display_server='http://localhost',
        name='test',
        checkpoints_dir='.',
        save_log=True,
        display_freq=1,
        print_freq=1,
        distinct_env=False,
    )
    opt = AttrDict(opt)
    vis = Visualizer(opt)
    vis.visualize(10, 10, 3, 90, {'a': 5., 'b': 10}, {'aaa': torch.randn(3, 540, 720), 'bbb': torch.randn(3, 540, 720)})
