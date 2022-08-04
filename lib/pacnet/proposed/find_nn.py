

# -- original --
import numpy as np
import torch as th
from einops import rearrange
import torch.nn.functional as nn_func
from .auxiliary_functions import reflection_pad_3d

# -- misc -
from pacnet.utils import optional

# -- proposed in-place CUDA search module --
import dnls
from dnls.utils import get_nums_hw

def run(vid,flows,k=10,ps=15,pt=1,ws=10,wt=0,dilation=1,stride0=1,stride1=1,
        ps_f=7,use_k=True,reflect_bounds=True,reshape_output=True):
    """
    The proposed nn search.

    """

    # -- include padding --
    in_vshape = vid.shape
    pad_r = ps_f + ps_f//2
    print(pad_r)
    pad_c = ws//2
    print("[pre] vid.shape: ",vid.shape,ws,ps)
    pad = [pad_r,]*4
    vid = nn_func.pad(vid, pad, mode='reflect')
    pad = [pad_c,]*4
    vid = nn_func.pad(vid, pad, mode='constant',value=-1.)
    print("[pad] vid.shape: ",vid.shape,ws,ps,pad_r,pad_c)

    # -- unpack --
    t,c,h,w = vid.shape
    fflow = None#optional(flows,'fflow',None)
    bflow = None#optional(flows,'bflow',None)

    # -- init search --
    h0_off,w0_off = 0,0
    h1_off,w1_off = 0,0
    use_adj,search_abs,exact = False,False,False
    search = dnls.search.init("l2_with_index",
                              fflow, bflow, k, ps, pt,
                              ws, wt, dilation=dilation,
                              stride0=stride0, stride1=stride1,
                              use_k = use_k,use_adj=use_adj,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,exact=exact,
                              h0_off=h0_off,w0_off=w0_off,
                              h1_off=h1_off,w1_off=w1_off)

    # -- batch searching info --
    nh,nw = get_nums_hw(vid.shape,stride0,ps,dilation,
                        pad_same=False,only_full=False)
    ntotal = t * nh * nw

    # -- batchsize info --
    MAX_NBATCH = 128*1024
    # nbatch = ntotal
    nbatch = 2*1024
    nbatch = min(ntotal,MAX_NBATCH)
    nbatches = (ntotal - 1)//nbatch + 1

    # -- run for batches --
    dists,inds = [],[]
    for batch_index in range(nbatches):

        # -- get batch info --
        qindex = min(nbatch * batch_index,ntotal)
        nbatch_i =  min(nbatch, ntotal - qindex)

        # -- run search --
        dists_i,inds_i = search(vid,qindex,nbatch_i)
        dists_i /= c

        # -- accumulate --
        dists.append(dists_i)
        inds.append(inds_i)

    # -- group --
    dists = th.cat(dists)
    inds = th.cat(inds)

    # -- for testing --
    # print("inds.shape: ",inds.shape) # 128 + 6 = 134
    if reshape_output:
        dists,inds = reshape_nn_pair(dists,inds,vid.shape)
        t,c,h,w = in_vshape
        c_s = pad_c + ps_f #pad_r#+pad_c
        c_s = (c_s-ps_f//2) if use_adj else c_s
        h_e,w_e = c_s + h + 2*(ps_f//2),c_s + w + 2*(ps_f//2)
        dists = dists[:,c_s:h_e,c_s:w_e,:]
        inds = inds[:,c_s:h_e,c_s:w_e,:,:]

    return dists,inds

def reshape_nn_pair(dists,inds,vshape):
    t,c,h,w = vshape
    dists = rearrange(dists,'(t h w) k -> t h w k',h=h,w=w)
    inds = rearrange(inds,'(t h w) k tr -> t h w k tr',h=h,w=w)
    return dists,inds

