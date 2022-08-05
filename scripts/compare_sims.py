"""
Compare similar images from PaCNet's Original and Our Proposed Search Methods

"""

# -- misc --
import os,math,tqdm
import pprint,random
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- network --
import pacnet
import pacnet.configs as configs
from pacnet import lightning
from pacnet.utils.misc import optional
import pacnet.utils.gpu_mem as gpu_mem
from pacnet.utils.misc import rslice,write_pickle,read_pickle
from pacnet.utils.metrics import compute_ssims,compute_psnrs


def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- set seed --
    configs.set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.sims_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_sims = []
    results.timer_flow = []
    results.mem_res = []
    results.mem_alloc = []

    # -- load model with sim fxn --
    model = pacnet.get_deno_model(cfg.model_name,cfg)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]

    # -- optional filter --
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",0)
    if frame_start >= 0 and frame_end > 0:
        def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
        indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
                                               cfg.frame_start,cfg.frame_end)]
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data.te[index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums']
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- divide --
        noisy /= 255.
        clean /= 255.

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        nframes = noisy.shape[0]
        print("(post-crop) [%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = pacnet.utils.timer.ExpTimer()

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            noisy_np = noisy.cpu().numpy()
            if noisy_np.shape[1] == 1:
                noisy_np = np.repeat(noisy_np,3,axis=1)
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

        # -- denoise --
        timer.start("sims")
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            sims = model.vid_cnn.compute_sims(noisy,clean,flows)
        timer.stop("sims")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        sims_s = sims[:,0,:3,:,:]
        sims_fns = pacnet.utils.io.save_burst(sims_s,out_dir,"sims")

        # -- noisy psnr [checking] --
        noisy_psnrs = compute_ssims(noisy,clean,div=1.)
        print(noisy.max(),clean.max())

        # -- sim psnr --
        t,c = noisy.shape[:2]
        ps_f = optional(cfg,'ps_f',7)
        k = optional(cfg,'k',15)
        nimgs = ps_f*ps_f
        psnrs = np.zeros((t,k-1,nimgs))
        ssims = np.zeros((t,k-1,nimgs))
        print("sims.shape: ",sims.shape)
        print("clean.shape: ",clean.shape)
        for ki in range(k-1):
            for ii in range(nimgs):
                cs = slice(ii*c,(ii+1)*c)
                ssims[:,ki,ii] = compute_ssims(clean,sims[ki+1,:,cs],div=1.)
                psnrs[:,ki,ii] = compute_psnrs(clean,sims[ki+1,:,cs],div=1.)
        print("-"*30)
        print(ssims.mean(2).mean(0))
        print(psnrs.mean(2).mean(0))
        print("-"*30)

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.sims_fns.append(sims_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)

    return results


def main():
    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "compare_sims"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    # cfg.isize = "96_96"
    # cfg.isize = "460_460"
    cfg.isize = "256_256"
    # cfg.isize = "none"
    cfg.bw = False
    cfg.nframes = 4
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.pacnet_ngpus = 1
    cfg.pacnet_time_chunk = 4

    # -- get mesh --
    ws,wt,k,bs,stride = [21],[4],[15],[32*1024],[1]
    dnames,sigmas,use_train = ["set8"],[50.],["false"]
    vid_names = ["sunflower"]
    # vid_names = ["tractor"]
    # vid_names = ["sunflower","hypersmooth","tractor"]
    # vid_names = ["snowboard","sunflower","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    ps_f,ps_s = [7],[15]
    flow,isizes,adapt_mtypes = ["true"],["none"],["rand"]
    model_names = ["proposed"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "isize":isizes,"use_train":use_train,"stride":stride,
                 "ws":ws,"wt":wt,"k":k, "bs":bs, "model_name":model_names,
                 'ps_f':ps_f,'ps_s':ps_s}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two

    # -- original w/out training --
    exp_lists['ps_f'] = [7]
    exp_lists['ps_s'] = [15]
    exp_lists['ws'] = [35]
    exp_lists['model_name'] = ["original"]
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exp_lists['ca_fwd'] = ["default"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a + exps_b
    # exps = exps_b# + exps_b

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # if exp.model_name == "proposed":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    for mname,mdf in records.groupby("model_name"):
        print(mname)
        psnrs = np.stack(mdf['psnrs'].to_numpy())[0,0]
        print(psnrs.mean(0).mean(-1))

if __name__ == "__main__":
    main()
