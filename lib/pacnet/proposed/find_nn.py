

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
        add_padding=True,bs=None,tstart=None,tend=None):
    """
    The proposed nn search.

    """

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
    fflow,bflow = compute_flows(vid)#flowd

    # -- init search --
    h0_off,w0_off = 0,0
    h1_off,w1_off = 0,0
    use_adj,search_abs,exact = False,False,False
    ws = ws if use_prop_nn else 10
    ps = 11
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

    # -- batchsize info --
    MAX_NBATCH = min(128*1024,ntotal)
    nbatch = 128*1024 if bs is None else bs
    nbatch = min(nbatch,MAX_NBATCH)
    nbatches = (ntotal - 1)//nbatch + 1
    # print(th.cuda.memory_reserved()/(1024.**3))
    # print(th.cuda.memory_allocated()/(1024.**3))
    # print(ws,wt,k,ps,pt,nbatch,qstart,qend,tstart,tend)

    # -- run for batches --
    qindex = qstart
    dists,inds = [],[]
    for batch_index in range(nbatches):
        if (batch_index+1) % 25 == 0:
            print("%d/%d" % (batch_index+1,nbatches))

        # -- get batch info --
        nbatch_i =  min(nbatch, qend - qindex)

        # -- run search --
        dists_i,inds_i = search(vid,qindex,nbatch_i)
        dists_i /= c

        # -- accumulate --
        dists.append(dists_i.cpu())
        inds.append(inds_i.cpu())

        # -- update --
        qindex += nbatch_i

    # -- group --
    dists = th.cat(dists).to(device)
    inds = th.cat(inds).to(device)

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

def compute_flows(vid):
    import svnlb
    device = vid.device
    vid_np = vid.cpu().numpy()
    if vid_np.shape[1] == 1:
        vid_np = np.repeat(vid_np,3,axis=1)
    flows = svnlb.compute_flow(vid_np,50.)
    fflow = th.from_numpy(flows.fflow).to(device)
    bflow = th.from_numpy(flows.bflow).to(device)
    return fflow,bflow

