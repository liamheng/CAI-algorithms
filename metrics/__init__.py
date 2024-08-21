from utility.configuration import CONFIG
from utility.file_locating import obtain_class_from_file
from metrics.base_metric import BaseMetrics

class Metrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.metrics = CONFIG['EVAL_METRICS']
        self.metrics = {m: obtain_class_from_file(f'metrics/{m}.py', m)() for m in CONFIG['EVAL_METRICS']}

    def __call__(self, preds, gts):
        return {k: v(preds, gts) for k, v in self.metrics.items()}
    
METRICS = Metrics()