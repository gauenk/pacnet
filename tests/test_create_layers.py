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

def relative_compare(tensor_gt,tensor_te,tol_mean,tol_max,tol_big,
                     eps_big=10,eps_small=1e-5,eps_smooth=1e-10):
    diff = th.abs(tensor_gt - tensor_te)/(tensor_gt.abs()+eps_smooth)
    args_big = th.where(diff > eps_big)
    args_check = th.where(diff < eps_big)
    args = th.where(tensor_gt.abs() > eps_small)

    # -- check not too many are big --
    nbig_perc = len(args_big[0])/(1.*tensor_gt.numel())
    print(nbig_perc,tol_big)
    assert nbig_perc < tol_big

    # -- compare pix values of remaining ones --
    error = diff[args_check].mean().item()
    assert error < tol_mean

    error = diff[args_check].max().item()
    assert error < tol_max

def test_original_proposed(sigma,ref_version):
    """

    Our test must tolerate different patch values at the same location.
    Different patch content can result in the same "dist".
    We expect this to be uncommon and use this in our test
    """

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
    fn = Path("create_layers.pt")
    recomp = True
    if not fn.exists() or recomp:
        if fn.exists(): os.remove(str(fn))
        nsearch = 15
        dists_gt,inds_gt = pacnet.original.find_nn.run(vid,nsearch)
        sim_gt = pacnet.original.create_layers.run(vid,inds_gt,nsearch)
        th.save([dists_gt,inds_gt,sim_gt],str(fn))
    else:
        triplet = th.load(str(fn))
        dists_gt,inds_gt,sim_gt = triplet[0],triplet[1],triplet[2]
    sim_gt = sim_gt[...,0,:,:]
    print("sim_gt.shape: ",sim_gt.shape)
    print("inds_gt.shape: ",inds_gt.shape)
    # dists_gt = dists_gt[:,3:-3,3:-3]
    # inds_gt = inds_gt[:,3:-3,3:-3]
    # sim_gt_s = sim_gt[0,0,:,:3,:,:]
    # print("sim_gt_s.shape: ",sim_gt_s.shape)
    # dnls.testing.data.save_burst(sim_gt_s,"./output/tests/find_nn/","sim0")
    # sim_gt_s = sim_gt[0,0,:,3:6,:,:]
    # dnls.testing.data.save_burst(sim_gt_s,"./output/tests/find_nn/","sim1")
    # sim_gt_s = sim_gt[0,0,:,-3:,:,:]
    # dnls.testing.data.save_burst(sim_gt_s,"./output/tests/find_nn/","sim-1")
    # sim_gt_s = th.abs(sim_gt[0,0,:,-3:,:,:] - sim_gt[0,0,:,:3,:,:])
    # sim_gt_s /= sim_gt_s.max()
    # dnls.testing.data.save_burst(sim_gt_s,"./output/tests/find_nn/","sim_diff")
    # exit(0)

    # -- proposed --
    k = 15#15*15*2
    dists_te,inds_te = pacnet.proposed.find_nn.run(vid,flows,k=k,ws=15,wt=1,ps=15,
                                                   pt=1,dilation=1,stride0=1,stride1=1,
                                                   use_k=True,reflect_bounds=False)
    inds_tx = convert_inds(vid,inds_gt,inds_te)
    # inds_tx = inds_te
    # print(inds_te[0,0,0,:3])
    # print(inds_tx[0,0,0,:3])
    # print("inds_te.shape: ",inds_te.shape)
    # print("inds_tx.shape: ",inds_tx.shape)
    # exit(0)
    sim_te = pacnet.proposed.create_layers.run(vid,inds_tx)[...,0,:,:]
    print("sim_te.shape: ",sim_te.shape)
    sim_te_s = sim_te[0,0,:,:3,:,:]
    print("sim_te_s.shape: ",sim_te_s.shape)
    dnls.testing.data.save_burst(sim_te_s,"./output/tests/find_nn/","sim_te")

    print("sim_gt.shape: ",sim_gt.shape)
    print("sim_te.shape: ",sim_te.shape)
    # dists_te = dists_te[...,1:]
    # inds_te = inds_te[...,1:,:]

    # -- viz sim --
    # print(inds_tx[0,0,0,:3])
    # print(sim_gt[0,0,0,:,0,0].view(7,7,3).permute(2,0,1))
    # print(sim_te[0,0,0,:,0,0].view(7,7,3).permute(2,0,1))
    # print("-"*20)
    # print("-"*20)
    # print(sim_gt[0,0,:,0,0,0])
    # print(sim_te[0,0,:,0,0,0])
    # print("-"*20)
    # print("-"*20)
    # print(sim_gt[0,0,0,0,:5,:5])
    # print(sim_te[0,0,0,0,:5,:5])
    # print("-"*20)
    # print(sim_gt[0,0,0,4,:5,:5])
    # print(sim_te[0,0,0,4,:5,:5])
    # print("-"*20)
    # print(sim_gt[0,0,0,5,:5,:5])
    # print(sim_te[0,0,0,5,:5,:5])
    # print("-"*20)
    # print(sim_gt[0,0,0,6,:5,:5])
    # print(sim_te[0,0,0,6,:5,:5])
    # print("-"*20)
    # print(sim_gt[0,0,0,7,:5,:5])
    # print(sim_te[0,0,0,7,:5,:5])
    # print("-"*20)
    # exit(0)

    # -- sim @ 0th channel; 0:3 --
    # sim_gt0 = sim_gt[:,0,:,:3]
    # sim_te0 = sim_te[:,0,:,:3]
    # diff = th.abs(sim_gt0 - sim_te0)/(sim_gt0.abs()+1e-10)
    # print("diff.shape: ",diff.shape)
    # print(diff.mean(),diff.max())

    # -- sim @ 1 --
    print("sim_gt.shape: ",sim_gt.shape)
    # print(inds_tx.shape)
    sim_gt1 = sim_gt[0,0,1,:,0,32]
    sim_te1 = sim_te[0,0,1,:,0,32]
    print(th.abs(sim_gt1 - sim_te1).sum())
    print(th.where(th.abs(sim_gt1 - sim_te1)>1e-3))
    print(sim_gt1)
    print(sim_te1)
    # exit()
    # diff = th.abs(sim_gt1 - sim_te1)/(sim_gt1.abs()+1e-10)
    # print("diff.shape: ",diff.shape)
    # print(diff.mean(),diff.max())

    # -- viz diff --
    diff = th.abs(sim_gt - sim_te)/(sim_gt.abs()+1e-10)
    # print(diff.shape)
    print(diff.mean(),diff.max())
    args = th.where(diff > 100.)
    # print(args)
    print("frames: ",th.unique(args[0]))
    print("neighs: ",th.unique(args[2]))
    print("channels: ",th.unique(args[3]))
    print("rows: ",th.unique(args[4]))
    print("cols: ",th.unique(args[5]))
    # print(diff[args])
    print(sim_gt[args][:3])
    print(sim_te[args][:3])
    diff[args] = 1
    diff = th.mean(diff,3)
    diff = diff[:,0,:,None,:,:]
    print(diff.mean(),diff.max())
    print("diff.shape: ",diff.shape)
    diff /= diff.max()
    dnls.testing.data.save_burst(diff[0],"./output/tests/find_nn/","dists0_k")
    dnls.testing.data.save_burst(diff[1],"./output/tests/find_nn/","dists1_k")

    # -- compare sim images --
    tol_mean,tol_max,tol_big = 1e-6,5e-5,0.03
    relative_compare(sim_gt,sim_te,tol_mean,tol_max,tol_big,
                     eps_big=1,eps_small=1e-5,eps_smooth=1e-10)

def convert_inds(vid,inds,inds_te):
    """
    inds_gt -> inds_te
    """

    #
    # -- compute 3d inds --
    #

    print("inds.shape: ",inds.shape)
    print("inds_te.shape: ",inds_te.shape)
    # inds = inds[:,3:-3,3:-3,:]
    # -- padding --
    # pad_s = (nsearch-1)//2
    # pad_r = 10
    # pad = pad_s + pad_r
    ws = 15
    ps = 7
    pad_r = ps//2 + ps
    pad_c = ws//2
    t_pad = 3

    # -- pad vid size --
    t,c,h,w = vid.shape
    hp,wp = h + 2*pad_r,w+2*pad_r
    print("hp,wp: ",hp,wp)

    # print(inds[0,0,0,0])
    print(inds[0,0,0,:])
    print(inds[1,0,0,:])

    inds_3d = []
    for ti in range(t):
        inds_i = inds[ti]
        inds_i -= max(t_pad-ti,0)*hp*wp # padded t
        # print(">0? ",th.all(inds_i>=0))
        tids = th.roll(th.arange(0,t),ti)
        # print(tids)
        inds_t = inds_i // (hp*wp)
        # for tj in range(t):
        #     inds_t[tj] = tids[tj]
        inds_mod = th.remainder(inds_i,hp*wp)
        inds_h = inds_mod // wp
        inds_w = inds_mod % wp
        inds_3d_t = th.stack([inds_t,inds_h,inds_w],-1)
        inds_3d_t = inds_3d_t.int()
        inds_3d.append(inds_3d_t)
    inds_3d = th.stack(inds_3d)
    # exit(0)

    # -- final add for padding --
    inds_3d[...,1] += ps
    inds_3d[...,2] += ps

    # -- add self --
    print("inds_te.shape: ",inds_te.shape)
    print("inds_3d.shape: ",inds_3d.shape)
    inds_3d = th.cat([inds_te[...,[0],:],inds_3d],-2)
    print("inds_3d.shape: ",inds_3d.shape)
    # print(inds_3d[0,0,0,:3])
    # print(inds_3d[0,1,1,:3])
    print("-"*10)
    # print(inds_3d[1,0,0,:3])
    # print(inds_te[1,0,0,:3])

    print(inds_3d[1,0,121,:3])
    print(inds_te[1,0,121,:3])

    print("-"*10)
    print(inds_3d[1,1,1,:6])
    print(inds_te[1,1,1,:6])
    print("-"*10)


    return inds_3d
