from schedulers.base_scheduler import BaseScheduler
import importlib

def find_scheduler(scheduler_name):
    
    # import py module of scheduler
    scheduler_filename = "schedulers.%s_scheduler" % scheduler_name
    scheduler_lib = importlib.import_module(scheduler_filename)

    scheduler = None

    # get scheduler
    scheduler_classname = scheduler_name.split(".")[-1].replace('_', '') + 'scheduler'
    for name, cls in scheduler_lib.__dict__.items():
        if name.lower() == scheduler_classname.lower() and issubclass(cls, BaseScheduler):
            scheduler = cls
    if scheduler is None:
        print("In %s.py, there should be a subclass of BaseScheduler with class name that matches %s in lowercase." % (scheduler_filename, scheduler_classname))
        exit(0)
    
    return scheduler