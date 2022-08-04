# -- pub api --
from . import aaai23

# -- code api --
from . import original
from . import proposed
from . import configs
from . import lightning
from . import utils
from .utils.misc import optional

def get_model(model_name,mtype,cfg=None):
    if model_name == "original":
        model = original.load_model(mtype,cfg).to(device)
        return model
    elif model_name == "proposed":
        model = proposed.load_model(mtype,cfg).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")

def get_deno_model(model_name,cfg):
    device = optional(cfg,"device","cuda:0")
    if model_name == "original":
        model = original.load_model("denoising",cfg).to(device)
        return model
    elif model_name == "proposed":
        model = proposed.load_model("denoising",cfg).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")

