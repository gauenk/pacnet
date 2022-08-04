import numpy as np
import torch as th
from einops import rearrange
import torch.nn.functional as nn_func
from .auxiliary_functions import reflection_pad_3d
from torch.nn.functional import unfold,fold
import dnls

def run(vid, inds, nsearch=15, ps_s=15, ps_f=7):

    # -- reflect --
    t,c,h,w = vid.shape
    # pad = (nsearch//2) + ps_f//2
    # pad_full = pad + nsearch//2
    print(vid.shape)
    pad_r = ps_f//2 + ps_f
    pad_c = (nsearch-1)//2
    vid = nn_func.pad(vid, [pad_r,]*4, mode='reflect')
    vid = nn_func.pad(vid, [pad_c,]*4, mode='constant',value=-1)
    vid = vid[...,4:-4,4:-4].contiguous()
    pad_diff = 4
    print("[new pad] vid.shape: ",vid.shape)

    # -- diff of inds --
    inds[...,1] = inds[...,1] - pad_diff# + 2
    inds[...,2] = inds[...,2] - pad_diff# + 2
    # print("MUST BE!: ",th.all(inds>=0))
    # print("MUST BE!: ",th.any(inds==0))
    # exit(0)

    print(inds[0,0,0,:7])
    print(inds[1,0,0,:7])
    print("vid.shape: ",vid.shape)
    print("inds.shape: ",inds.shape)

    # -- init unfold --
    pt,dil = 1,1
    adj,r_bounds = 0,False
    unfold_k = dnls.UnfoldK(ps_f,pt=pt,dilation=dil,
                            adj=adj,reflect_bounds=r_bounds)
    # -- unfold --
    inds = rearrange(inds,'t h w k tr -> (t h w) k tr')
    print("inds.shape: ",inds.shape)
    # inds = inds[:,1:] # remove self
    patches = unfold_k(vid,inds)
    # px = 13
    # print(vid[0,:,px-3:px+4,px-3:px+4].transpose(2,1))
    print("patches.shape: ",patches.shape)
    # # print(patches[0,:7,0,0,0,0])
    # print(patches[0,0,0].transpose(2,1))

    # -- mangle patches --
    # patches = rearrange(patches,'b k pt c ph pw -> k b 1 1 (pt ph pw c) 1 1')
    # sim_layers = rearrange(patches,'k (t h w) 1 1 c 1 1 -> t 1 k c 1 h w',h=h,w=w)
    # print("sim_layers.shape: ",sim_layers.shape)
    # exit(0)
    # t,c,hp,wp = vid.shape
    hp,wp = h + 2*(ps_f//2),w + 2*(ps_f//2)

    ps = ps_f
    patches = rearrange(patches,'(t h w) k pt c ph pw -> t h w k pt c ph pw',
                        h=hp,w=wp)
    nh,nw = (h-1)//ps + 1,(w-1)//ps + 1
    k = patches.shape[3]

    pdim = ps*ps*c
    sim_h,sim_w = ps*nh,ps*nw
    sim_vids = th.zeros((k,t,pdim,sim_h+ps,sim_w+ps),device=vid.device)
    sim_vids[...] = th.nan
    # print("sim_vids.shape: ",sim_vids.shape)
    c0 = 0
    for pi in range(ps):
        pi_end = pi+nh*ps+1
        for pj in range(ps):
            pj_end = pj+nw*ps+1
            patches_ij = patches[:,pi:pi_end:ps,pj:pj_end:ps]
            # print("patches_ij.shape: ",patches_ij.shape)
            shape_str = 't h w k 1 c ph pw -> k t c (h ph) (w pw)'
            tmp = rearrange(patches_ij,shape_str)
            # print("tmp.shape: ",tmp.shape)
            # print("a,b: ",pi+sim_h,pj+sim_w)
            _,_,_,h_ij,w_ij = tmp.shape
            sim_vids[:,:,c0:c0+3,pi:pi+h_ij,pj:pj+w_ij] = tmp[...]
            c0 += 3
    # print(sim_vids.shape)
    ps2 = 2*(ps_f//2)
    sim_vids = sim_vids[...,ps2:-ps2,ps2:-ps2]
    # print(ps2)
    # args = th.where(th.isnan(sim_vids))
    # print(args)

    # args = th.where(th.isnan(sim_vids))
    # print(args)
    # print("any nan?: ",th.any(th.isnan(sim_vids)))
    # exit(0)

    # print(sim_vids.shape)
    # exit(0)
    sim_vids = rearrange(sim_vids,'k t c h w -> t 1 k c 1 h w')

    return sim_vids
