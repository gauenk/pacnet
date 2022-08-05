
# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- model --
from .modules import PaCNet

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
    device = optional(cfg,"device","cuda:0")
    ntype =  optional(cfg,"ntype","gaussian")
    sigma =  optional(cfg,"sigma",50.)
    read_noise =  optional(cfg,"read_noise",10.)
    shot_noise =  optional(cfg,"shot_noise",25.)
    qis_lambda =  optional(cfg,"qis_lambda",20.)
    a,b = get_default_deno_states(cfg)
    default_vidcnn_state_fn = a
    default_tfnet_state_fn = b

    # -- params --
    device = optional(cfg,'device','cuda:0')
    vidcnn_state_fn =  optional(cfg,"vidcnn_state_fn",default_vidcnn_state_fn)
    tfnet_state_fn =  optional(cfg,"tfcnn_state_fn",default_tfnet_state_fn)
    ws = optional(cfg,'ws',29)
    wt = optional(cfg,'wt',3)
    k = optional(cfg,'k',15)
    pacnet_time_chunk = optional(cfg,'pacnet_time_chunk',3)
    ps_s = optional(cfg,'ps_s',15)
    ps_f = optional(cfg,'ps_f',7)
    ngpus = optional(cfg,'pacnet_ngpus',2)
    max_gpus = optional(cfg,'max_gpus',3)

    # -- batch sizes per gpu --
    assert ngpus <= max_gpus
    bs = optional(cfg,'bs',32*1024)
    bs_list = []
    for gpuid in range(ngpus):
        bs_list.append(bs)

    # -- declare --
    model = PaCNet(ws=ws,wt=wt,k=k,ps_f=ps_f,ps_s=ps_s,
                   ngpus=ngpus,nn_bs_list=bs_list,
                   time_chunk=pacnet_time_chunk)

    # -- load state --
    vidcnn_state = th.load(vidcnn_state_fn)
    model.vid_cnn.load_state_dict(vidcnn_state['state_dict'])
    tfnet_state = th.load(tfnet_state_fn)
    model.tf_net.load_state_dict(tfnet_state['state_dict'])

    # -- eval --
    model.eval()
    model = model.to(device)

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

def get_default_deno_states(cfg):

    # -- unpack identifying info --
    ntype =  optional(cfg,"ntype","gaussian")
    sigma =  optional(cfg,"sigma",50.)
    read_noise =  optional(cfg,"read_noise",10.)
    shot_noise =  optional(cfg,"shot_noise",25.)
    qis_lambda =  optional(cfg,"qis_lambda",20.)
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"

    if ntype in ["gaussian","g"]:
        model_sigma = select_sigma(sigma)
        scnn_base = './weights/s_cnn_video/model_state_sig{}.pt'.format(model_sigma)
        s_cnn_state_fn = fdir/scnn_base
        tfnet_base = './weights/t_cnn/model_state_sig{}.pt'.format(model_sigma)
        tf_net_state_fn = fdir/tfnet_base
    elif ntype in ["qis"]:
        raise NotImplementedError("No qis default yet.")
    else:
        raise ValueError(f"Uknown noise type [{ntype}]")
    assert os.path.isfile(s_cnn_state_fn)
    assert os.path.isfile(tf_net_state_fn)
    return s_cnn_state_fn,tf_net_state_fn

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

