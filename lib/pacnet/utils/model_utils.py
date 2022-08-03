import os
import torch as th
import numpy as np
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    th.save(state, model_out_path)

def select_sigma(sigma):
    sigmas = np.array([10, 20, 30, 40, 50])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def load_checkpoint(model, weights):
    checkpoint = th.load(weights)
    try:
        # model.load_state_dict(checkpoint["state_dict"])
        raise ValueError("")
    except Exception as e:
        state_dict = checkpoint["net"]
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] if 'module.' in k else k
        #     new_state_dict[name] = v
        model.load_state_dict(state_dict)

def load_checkpoint_multigpu(model, weights):
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = th.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = th.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_model_weights(fdir,data_sigma,ntype):
    path = Path(fdir) / "weights"
    if ntype == "gaussian":
        path /= "results_gaussian_denoising"
        model_sigma = select_sigma(data_sigma)
        mdir = "pretrained_sigma%d" % model_sigma
        mdir_full = path / mdir / "checkpoint" / "051_ckpt.t7"
    elif ntype == "poisson":
        path /= "results_poissongaussian_denoising"
        mdir = "pretrained"
        mdir_full = path / mdir / "checkpoint" / "051_ckpt.t7"
    else:
        raise ValueError(f"Uknown noise type [{ntype}]")
    return str(mdir_full)


