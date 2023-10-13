from models.IDCGAN_model import IDCGANModel

def get_option_setter():
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = IDCGANModel
    return model_class.modify_commandline_options


def create_model(opt):
    model = IDCGANModel
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
