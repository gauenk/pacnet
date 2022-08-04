"""

Test versions of Pacnet to differences in output due to code modifications.

"""

# -- misc --
import sys,os,tqdm,pytest,math,random
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import dnls # supporting
from torchvision.transforms.functional import center_crop
import torch.nn.functional as nn_func

# -- package imports [to test] --
import pacnet
from pacnet.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats
from pacnet.utils.misc import rslice_pair

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_denose_rgb/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # th.use_deterministic_algorithms(True)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"sigma":[50.],"ref_version":["ref","original"]}
    test_lists = {"sigma":[50.],"ref_version":["ref"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# -->  Test original vs proposed code base  <--
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def relative_compare(tensor_gt,tensor_te,tol_mean,tol_max,
                     eps_small=1e-5,eps_smooth=1e-10):
    diff = th.abs(tensor_gt - tensor_te)/(tensor_gt.abs()+eps_smooth)
    args = th.where(tensor_gt.abs() > eps_small)

    error = diff.mean().item()
    assert error < tol_mean

    error = diff[args].max().item()
    assert error < tol_max

def test_original_proposed(sigma,ref_version):

    # -- params --
    device = "cuda:0"
    vid_set = "set8"
    vid_name = "motorbike"
    isize = "128_128"
    dset = "te"
    flow = False
    noise_version = "blur"
    verbose = True

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = vid_name
    cfg.isize = isize
    cfg.sigma = 30.
    cfg.bw = False
    cfg.nframes = 2
    cfg.seed = 123

    # -- video --
    th.manual_seed(cfg.seed)
    data,loaders = data_hub.sets.load(cfg)
    groups = data[dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    index = indices[0]

    # -- unpack --
    sample = data[dset][index]
    region = sample['region']
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = rslice_pair(noisy,clean,region)
    noisy,clean = noisy.to(device),clean.to(device)
    vid_frames = sample['fnums']
    noisy /= 255.
    vid = noisy

    # -- flows --
    t,c,h,w = noisy.shape
    flows = edict()
    flows.fflow = th.zeros((t,2,h,w),device=noisy.device)
    flows.bflow = th.zeros((t,2,h,w),device=noisy.device)
    hw = h*w

    # -- original --
    fn = Path("pair.pt")
    recomp = True
    if not fn.exists() or recomp:
        if fn.exists(): os.remove(str(fn))
        dists_gt,inds_gt = pacnet.original.find_nn.run(vid)
        th.save([dists_gt,inds_gt],str(fn))
    else:
        pair = th.load(str(fn))
        dists_gt,inds_gt = pair[0],pair[1]
    print("inds_gt.shape: ",inds_gt.shape)
    dists_gt = dists_gt[:,3:-3,3:-3]
    inds_gt = inds_gt[:,3:-3,3:-3]

    # -- proposed --
    k = 15#15*15*2
    dists_te,inds_te = pacnet.proposed.find_nn.run(vid,flows,k=k,ws=15,wt=1,ps=15,
                                                   pt=1,dilation=1,stride0=1,stride1=1,
                                                   use_k=True,reflect_bounds=False)
    # print("dists_gt.shape: ",dists_gt.shape)
    # print("dists_te.shape: ",dists_te.shape)
    # dists_te = dists_te[...,10:-10,10:-10,1:]
    dists_te = dists_te[...,1:]
    inds_te = inds_te[...,1:,:]
    # print(inds_gt[0,0,0,0])
    # print(inds_gt[0,0,0,1])

    # -- prints --
    # print("-"*15)
    # print(dists_gt[0,16:24,16:24,0])
    # print(dists_te[0,16:24,16:24,0])
    # print("-"*15)
    # print(dists_gt[0,16:24,16:24,1])
    # print(dists_te[0,16:24,16:24,1])
    # print("-"*15)
    # print(dists_gt[0,16:24,16:24,5])
    # print(dists_te[0,16:24,16:24,5])
    # print("-"*15)
    # print(dists_te[0,:3,:3,1])

    # -- viz inds --
    # print("-"*15)
    # print(inds_gt[0,0,0,:3])
    # print(inds_te[0,0,0,:3])

    # -- viz patches --
    # test_t,test_h,test_w = 0,10,12
    # patches_gt,pinds_gt = get_patches_at_k_original(inds_gt[test_t,test_h,test_w,None,:],vid)
    # patches_te,pinds_te = get_patches_at_k_proposed(inds_te[test_t,test_h,test_w,None,:],vid)
    # print(pinds_gt)
    # print(pinds_te[:,:14,:])
    # print(pinds_te[...,1].min(),pinds_te[...,1].max())
    # print(pinds_te[...,2].min(),pinds_te[...,2].max())

    # test_ind = th.IntTensor([[[1,21,15]]]).to(pinds_te.device)
    # diff = th.sum(th.abs(pinds_te - test_ind),-1)
    # print(diff.shape)
    # print(diff)
    # iargs = th.where(diff<1e-10)
    # print(iargs)
    # print(pinds_te[iargs])


    # for i in range(15):
    #     test_ind = (20 - 7) + i
    #     diff = th.abs(pinds_te[...,1] - test_ind)
    #     # print(diff.shape)
    #     # print(diff)
    #     iargs = th.where(diff<1e-10)
    #     # print(iargs)
    #     # print(pinds_te[iargs])
    #     print(test_ind,len(pinds_te[iargs]))

    # test_ind = 21
    # diff = th.abs(pinds_te[...,1] - test_ind)
    # iargs = th.where(diff<1e-10)
    # print(test_ind,len(pinds_te[iargs]))
    # sinds = pinds_te[iargs]
    # order_a = th.argsort(sinds[...,0])
    # sinds = sinds[order_a]
    # order_b = th.argsort(sinds[...,-1])
    # sinds = sinds[order_b]
    # print(sinds)
    # print(len(sinds))

    # exit(0)

    # print(iargs)
    # print(pinds_te[...,0][iargs])
    # print(pinds_te[...,1][iargs])
    # print(pinds_te[...,2][iargs])

    # print(patches_gt.shape)
    # print(patches_te.shape)
    # patches_gt = patches_gt.view(14,-1)
    # patches_te = patches_te.view(k,-1)

    # ps = 15
    # ps2 = ps*ps
    # diff_gt = th.mean((patches_gt - patches_gt[[0]])**2,-1) *ps2
    # diff_te = th.mean((patches_te - patches_te[[0]])**2,-1) *ps2
    # print(diff_gt)
    # print(diff_te)

    # print(dists_gt[test_t,test_h,test_w])
    # print(dists_te[test_t,test_h,test_w])

    # print(diff_te[iargs[1][0]])
    # print(dists_te[test_t,test_h,test_w,iargs[1][0]])
    # exit(0)




    # patches_gt = get_patches_at_k_original(inds_gt[0,16,16,None,:],vid)
    # patches_te = get_patches_at_k_proposed(inds_te[0,16,16,None,:],vid)
    # print(patches_gt[0,0,0,0])
    # print(patches_te[0,0,0,0])

    # -- viz diff --
    diff = th.abs(dists_gt - dists_te)/(dists_gt.abs()+1e-10)
    print(diff.shape)
    print(diff.mean())
    print(diff.max())
    args = th.where(diff > 1e-2)
    print(args)
    print(diff[args])
    diff /= diff.max()
    diff = rearrange(diff,'t h w k -> t k 1 h w')
    dnls.testing.data.save_burst(diff[0],"./output/tests/find_nn/","dists0_k")
    dnls.testing.data.save_burst(diff[1],"./output/tests/find_nn/","dists1_k")

    # -- compare _only_ dists, "inds" can swap on equal dists --
    # print("dists_gt.shape: ",dists_gt.shape)
    # print("dists_te.shape: ",dists_te.shape)
    tol_mean,tol_max = 1e-6,5e-5
    relative_compare(dists_gt,dists_te,tol_mean,tol_max,
                     eps_small=1e-5,eps_smooth=1e-10)
    diff = th.abs(dists_gt - dists_te)/(dists_gt.abs()+1e-10)
    args = th.where(dists_gt.abs() < 1e-5)


def get_patches_at_k_original(inds,vid,nsearch=15):

    #
    # -- compute 3d inds --
    #

    # -- padding --
    # pad_s = (nsearch-1)//2
    # pad_r = 10
    # pad = pad_s + pad_r
    ps = 7
    pad = ps//2 + ps
    t_pad = 3

    # -- reflect --
    vid = nn_func.pad(vid, [pad,]*4, mode='reflect')

    # -- pad vid size --
    t,c,hp,wp = vid.shape
    # tp = t
    # hp = h + pad*2
    # wp = w + pad*2

    # -- unfold inds --
    # print(inds)
    inds -= t_pad*hp*wp # padded t
    inds_t = inds //(hp*wp)
    inds_mod = th.remainder(inds,hp*wp)
    inds_h = inds_mod // wp
    inds_w = inds_mod % wp
    inds_3d = th.stack([inds_t,inds_h,inds_w],-1)
    inds_3d = inds_3d.int()
    # print(inds)
    # print(inds_3d)
    # print(h,w,nsearch,hp,wp)

    # -- shift -- (top,left should be 10)
    inds_3d[:,:,1] = inds_3d[:,:,1]# - (ps + ps//2)
    inds_3d[:,:,2] = inds_3d[:,:,2]# - (ps + ps//2)

    # -- unfold patches --
    ps,pt,dil = 15,1,1
    adj,r_bounds = 0,True
    unfold_k = dnls.UnfoldK(ps,pt=pt,dilation=dil,adj=adj,reflect_bounds=r_bounds)
    patches = unfold_k(vid,inds_3d)
    return patches,inds_3d

def get_patches_at_k_proposed(inds,vid):

    # -- reflect --
    pad_full = 10 + 7
    pad = 10
    vid = nn_func.pad(vid, [10,]*4, mode='reflect')
    pad_diff = pad_full - pad

    # -- diff of inds --
    inds[...,1] = inds[...,1] - pad_diff
    inds[...,2] = inds[...,2] - pad_diff

    # -- unpack --
    ps,pt,dil = 15,1,1
    adj,r_bounds = 0,True
    unfold_k = dnls.UnfoldK(ps,pt=pt,dilation=dil,adj=adj,reflect_bounds=r_bounds)
    patches = unfold_k(vid,inds)
    return patches,inds
