from .general_options import GeneralOptions

class TestOptions(GeneralOptions):

    def __init__(self):
        super().__init__()
        self.define_options(self.option_configs["test"])
        self.parser.set_defaults(phase="test")