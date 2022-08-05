import numpy as np
import torch as th
from einops import rearrange
import torch.nn.functional as nn_func
from .auxiliary_functions import reflection_pad_3d
from torch.nn.functional import unfold,fold
import dnls

def run(vid, inds, nsearch=15, ps_s=15, ps_f=7, add_padding=True, testing=False):

    # -- reflect --
    pad_r = ps_f//2 + ps_f
    pad_c = (nsearch-1)//2
    pad_f = pad_r + pad_c
    # print(pad_r,pad_c)
    if add_padding:
        vid = nn_func.pad(vid, [pad_r,]*4, mode='reflect')
        vid = nn_func.pad(vid, [pad_c,]*4, mode='constant',value=-1)
        vid = vid[...,4:-4,4:-4].contiguous()
    t,c,hp0,wp0 = vid.shape
    h,w = hp0-2*(pad_f-4),wp0-2*(pad_f-4)
    # print("[new pad] vid.shape: ",vid.shape)
    # print("h,w: ",h,w,hp0,wp0)
    # print("inds.shape: ",inds.shape)

    # -- shift inds --
    inds[...,1] = inds[...,1] - 4
    inds[...,2] = inds[...,2] - 4
    # print(inds[:,0,0,:3])

    # -- init unfold --
    pt,dil = 1,1
    adj,r_bounds = 0,False
    unfold_k = dnls.UnfoldK(ps_f,pt=pt,dilation=dil,
                            adj=adj,reflect_bounds=r_bounds)
    # -- unfold --
    t,inds_h,inds_w,_,_ = inds.shape
    inds = rearrange(inds,'t h w k tr -> (t h w) k tr')
    patches = unfold_k(vid,inds)

    # -- mangle patches --
    patches = rearrange(patches,'(t h w) k pt c ph pw -> t h w k pt c ph pw',
                        h=inds_h,w=inds_w) # == img_h (no pads) + 2*(ps_f//2)


    # -- decl dim info --
    nh,nw = (h-1)//ps_f + 1,(w-1)//ps_f + 1
    # print("nh,nw: ",nh,nw)
    k = patches.shape[3]
    ps = ps_f
    pdim = ps*ps*c

    # -- decl endpoints --

    # -- init sim --
    sim_h = h + 2*(ps_f-1)
    sim_w = w + 2*(ps_f-1)
    # sim_h,sim_w = ps*nh,ps*nw
    sim_vids = th.zeros((k,t,pdim,sim_h,sim_w),device=vid.device)
    sim_vids[...] = th.nan
    # print("sim_vids.shape: ",sim_vids.shape)

    # inds_h = h - (nsearch - 1 + pad) # h - 80 = h - (nsearch - 1 + pad) = h - (75-1+6)
    # inds_w = w - (nsearch - 1 + pad)
    # edge_v = h - (nsearch-1) - pi
    # edge_h = w - (nsearch-1) - pj
    # # (h - 80 - (h - ps4 - map_v_s) % ps)
    # end_v = (inds_h - edge_v % ps)
    # end_h = (inds_w - edge_h % ps)

    # -- fill --
    c0 = 0
    for pi in range(ps):

        # -- endpoint --
        edge_i = hp0 - (nsearch-1) - pi
        pi_end = inds_h - edge_i % ps_f
        for pj in range(ps):

            # -- endpoint --
            edge_j = wp0 - (nsearch-1) - pj
            pj_end = inds_w - edge_j % ps_f
            # print(pi,pi_end,pj,pj_end,ps,edge_i,edge_j)

            # -- extract patches --
            patches_ij = patches[:,pi:pi_end:ps,pj:pj_end:ps]
            shape_str = 't h w k 1 c ph pw -> k t c (h ph) (w pw)'

            # -- fill sim --
            tmp = rearrange(patches_ij,shape_str)
            _,_,_,h_ij,w_ij = tmp.shape
            sim_vids[:,:,c0:c0+3,pi:pi+h_ij,pj:pj+w_ij] = tmp[...]

            # -- update channel --
            c0 += 3

    # -- extract center --
    ps2 = (ps_f-1)
    sim_vids = sim_vids[...,ps2:-ps2,ps2:-ps2]
    # print("Any nan? ",th.any(th.isnan(sim_vids)).item())

    # -- final reshape [testing only] --
    if testing:
        sim_vids = rearrange(sim_vids,'k t c h w -> t 1 k c 1 h w')

    return sim_vids
