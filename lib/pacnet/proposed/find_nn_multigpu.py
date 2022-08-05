"""

"""

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

def run(vid,flows,k=15,ps=15,pt=1,ws=10,wt=0,dilation=1,stride0=1,stride1=1,
        ps_f=7,use_k=True,reflect_bounds=True,use_prop_nn=True,reshape_output=True,
        add_padding=True,ngpus=2,bs_list=None,tstart=None,tend=None):
    """
    The proposed nn search.

    """
    print(bs_list)

    # -- include padding --
    device = vid.device # default device
    pad_r = ps_f + ps_f//2
    pad_c = ws//2
    pad_f = pad_r + pad_c
    # print("[pre] vid.shape: ",vid.shape,ws,ps)
    if add_padding:
        pad = [pad_r,]*4
        vid = nn_func.pad(vid, pad, mode='reflect')
        pad = [pad_c,]*4
        vid = nn_func.pad(vid, pad, mode='constant',value=-1.)
    # print("[pad] vid.shape: ",vid.shape,ws,ps,pad_r,pad_c)

    # -- compute shapes --
    t,c,hp,wp = vid.shape
    h,w = hp - 2*pad_f,wp - 2*pad_f

    # -- unpack --
    # print("h,w: ",h,w)
    fflow,bflow = get_flows(flows,vid.shape)

    # -- init search --
    h0_off,w0_off = 0,0
    h1_off,w1_off = 0,0
    use_adj,search_abs,exact = False,False,False
    ws = ws if use_prop_nn else 10
    print("info: ",ws,wt,ps,pt)
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
    tstart = 0 if tstart is None else tstart
    tend = t if tend is None else tend
    nt = tend - tstart
    ntotal = nt * nh * nw

    # -- get search end-points --
    qstart = tstart * nh * nw
    qend = ntotal + qstart
    # print("info: ",qstart,qend,tstart,tend)

    # -- init data on devices --
    if bs_list is None:
        bs_list = [128*1024,156*1024]#,48*1024]
    batch_sizes = bs_list
    vid_gpus = [vid.to(d) for d in get_devices(ngpus)]

    # -- batchsize info --
    nbatch = np.sum(batch_sizes).item()
    nbatch = min(nbatch,ntotal)
    nbatches = (ntotal - 1)//nbatch + 1

    # -- run for batches --
    qindex = qstart
    dists,inds = [],[]
    for batch_index in range(nbatches):

        # -- info --
        if (batch_index+1) % 25 == 0:
            print("%d/%d" % (batch_index+1,nbatches))

        # -- multi gpu id --
        for gpu_id in range(ngpus):

            # -- get batch info --
            nbatch_i = batch_sizes[gpu_id]
            nbatch_i = min(nbatch_i, qend - qindex)
            if nbatch_i == 0: continue

            # -- run search --
            vid_i = vid_gpus[gpu_id]
            dists_i,inds_i = search(vid_i,qindex,nbatch_i)
            dists_i /= c

            # -- accumulate --
            dists.append(dists_i.cpu())
            inds.append(inds_i.cpu())

            # -- incriment --
            qindex += nbatch_i

    # -- group --
    dists = th.cat(dists)#.to(device)
    inds = th.cat(inds)#.to(device)

    # -- for testing --
    # print("inds.shape: ",inds.shape) # 128 + 6 = 134
    if reshape_output:
        dists,inds = reshape_nn_pair(dists,inds,vid.shape)
        c_s = pad_c + ps_f #pad_r#+pad_c
        c_s = (c_s-ps_f//2) if use_adj else c_s
        h_e,w_e = c_s + h + 2*(ps_f//2),c_s + w + 2*(ps_f//2)
        dists = dists[:,c_s:h_e,c_s:w_e,:]
        inds = inds[:,c_s:h_e,c_s:w_e,:,:]
    # print(inds[0,0,0,:3])

    return dists,inds

def get_devices(ngpus):
    devices = []
    for index in range(ngpus):
        device = 'cuda:%d' % index
        devices.append(device)
    return devices

def get_device(index,ngpus):
    gpu_id = index % ngpus
    device = 'cuda:%d' % gpu_id
    return device

def reshape_nn_pair(dists,inds,vshape):
    t,c,h,w = vshape
    dists = rearrange(dists,'(t h w) k -> t h w k',h=h,w=w)
    inds = rearrange(inds,'(t h w) k tr -> t h w k tr',h=h,w=w)
    return dists,inds


def get_flows(flows,vshape):
    fflow = optional(flows,'fflow',None)
    bflow = optional(flows,'bflow',None)

    if not fflow is None:
        pad_h = (vshape[-2] - fflow.shape[-2])//2
        pad_w = (vshape[-1] - fflow.shape[-1])//2
        pads = [pad_h,pad_h,pad_w,pad_w]
        fflow = nn_func.pad(fflow, pads, mode='constant',value=0)
    if not bflow is None:
        pad_h = (vshape[-2] - bflow.shape[-2])//2
        pad_w = (vshape[-1] - bflow.shape[-1])//2
        pads = [pad_h,pad_h,pad_w,pad_w]
        bflow = nn_func.pad(bflow, pads, mode='constant',value=0)
    return fflow,bflow
