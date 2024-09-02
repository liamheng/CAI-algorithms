import argparse
import models
import loaders
import schedulers
import datetime
import json
import logging
from pydoc import locate


class GeneralOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.load_option_definitions()
        self.define_options(self.option_configs["general"])

    def define_options(self, option_configs):
 
        # add arguments
        for name, arguments in option_configs.items():
            self.parser.add_argument("--%s" % name, **arguments)
    
    def load_option_definitions(self):
        # import json file
        with open("static/options.json", 'r') as f:
            option_configs = json.load(f)
            
            # convert str to actual python data type
            for phase_configs in option_configs.values():
                for argument in phase_configs.values():
                    if "type" in argument:
                        argument["type"] = locate(argument["type"])
            self.option_configs = option_configs

    def initialize(self):
        opt, _ = self.parser.parse_known_args()

        # additional arguments from implemented model
        model_class = models.find_model(opt.model)
        self.parser = model_class.modify_commandline_options(self.parser, isTrain=opt.phase == "training")
        opt, _ = self.parser.parse_known_args()

        # additional arguments from implemented dataset
        dataset_class = loaders.find_dataset(opt.dataset)
        self.parser = dataset_class.modify_commandline_options(self.parser, isTrain=opt.phase == "training")
        opt, _ = self.parser.parse_known_args()

        if opt.phase == 'training':
            # additional arguments from implemented scheduler
            scheduler_class = schedulers.find_scheduler(opt.lr_policy)
            self.parser = scheduler_class.modify_commandline_options(self.parser)
            opt, _ = self.parser.parse_known_args()

        return self.parser.parse_args()


    def print_options(self, opt, time):
        # print options
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        logger = logging.getLogger(opt.name)
        logger.info("================ options recorded time: %s ================" % time)
        logger.info(message)




    def parse(self):
        opt = self.initialize()

        now = datetime.datetime.now()
        dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")

        # add timestamp to prevent overwrite
        if opt.phase == "training" and opt.add_timestamp:
            opt.name = opt.name + dt_string
        self.opt = opt
        return opt