
# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- model --
from .modules import VidCnn

# -- misc imports --
from pacnet.utils.misc import optional
from pacnet.utils.model_utils import load_checkpoint,select_sigma,get_model_weights

def load_model(mtype,cfg):
    if mtype in ["deno","denoising"]:
        return load_deno_model(cfg)
    elif mtype in ["sr","superres"]:
        return load_sr_model(cfg)
    else:
        raise ValueError(f"Uknown model to load [{mtype}]")

def load_deno_model(cfg):
    # -- load defaults --
    ntype =  optional(cfg,"ntype","gaussian")
    sigma =  optional(cfg,"sigma",50.)
    read_noise =  optional(cfg,"read_noise",10.)
    shot_noise =  optional(cfg,"shot_noise",25.)
    qis_lambda =  optional(cfg,"qis_lambda",20.)
    default_state_fn = get_default_deno_state(cfg)

    # -- params --
    device = optional(cfg,'device','cuda:0')
    state_fn =  optional(cfg,"model_state_fn",default_state_fn)

    # -- declare --
    model = VidCnn()

    # -- load state --
    model_state = th.load(state_fn)
    model.load_state_dict(model_state['state_dict'])
    model.eval()

    return model

def load_sr_model(cfg):

    # -- load defaults --
    sr_type =  optional(cfg,"sr_type","default")
    sr_scale =  optional(cfg,"sr_scale",4)
    default_state_fn = get_default_sr_state(cfg)

    # -- params --
    device = optional(cfg,'device','cuda:0')
    state_fn =  optional(cfg,"model_state_fn",default_state_fn)

    # -- declare --
    model = VidCnn()

    # -- load state --
    model_state = th.load(state_fn)
    model.load_state_dict(model_state['state_dict'])
    model.eval()

    return model

def get_default_deno_state(cfg):

    # -- unpack identifying info --
    ntype =  optional(cfg,"ntype","gaussian")
    sigma =  optional(cfg,"sigma",50.)
    read_noise =  optional(cfg,"read_noise",10.)
    shot_noise =  optional(cfg,"shot_noise",25.)
    qis_lambda =  optional(cfg,"qis_lambda",20.)
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"

    if ntype in ["gaussian","g"]:
        model_sigma = select_sigma(sigma)
        state_fn = fdir/'./weights/s_cnn_video/model_state_sig{}.pt'.format(model_sigma)
    elif ntype in ["qis"]:
        raise NotImplementedError("No qis default yet.")
    else:
        raise ValueError(f"Uknown noise type [{ntype}]")
    print(state_fn)
    assert os.path.isfile(state_fn)
    return state_fn

def get_default_sr_state(cfg):

    # -- unpack identifying info --
    sr_type =  optional(cfg,"sr_type","default")
    sr_scale =  optional(cfg,"sr_scale",4)
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"

    if sr_type == "default":
        state_fn = fdir/f'./weights/sr/{sr_scale}/model_state.pt'
    else:
        raise ValueError(f"Uknown super-res. type [{sr_type}]")
    print(state_fn)
    assert os.path.isfile(state_fn)
    return state_fn

