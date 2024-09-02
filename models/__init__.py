import importlib
from models.base_model import BaseModel

def find_model(model_name):
    
    # import py module of model
    model_filename = "models.%s_model" % model_name
    model_lib = importlib.import_module(model_filename)

    model = None

    # get model
    model_classname = model_name.split(".")[-1].replace('_', '') + 'model'
    for name, cls in model_lib.__dict__.items():
        if name.lower() == model_classname.lower() and issubclass(cls, BaseModel):
            model = cls
    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, model_classname))
        exit(0)
    
    return model

def create_model(opt, logger):

    model = find_model(opt.model)
    instance = model(opt)
    logger.info("[Create Model] model %s was created" % type(instance).__name__)
    return instance



