from .general_options import GeneralOptions

class TrainOptions(GeneralOptions):

    def __init__(self):
        super().__init__()
        self.define_options(self.option_configs["training"])

        