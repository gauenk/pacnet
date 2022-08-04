
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
    results.adapt_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_adapt = []
    results.timer_deno = []
    results.mem_res = []
    results.mem_alloc = []

    # -- network --
    model = pacnet.get_deno_model(cfg.model_name,cfg)
    model.eval()
    imax = 255.

    # -- optional load trained weights --
    load_trained_state(model,cfg.use_train)

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

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        print("(post-crop) [%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = pacnet.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = 390*39#ngroups*1024

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

        # -- internal adaptation --
        timer.start("adapt")
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        adapt_psnrs = [0.]
        if run_internal_adapt:
            adapt_psnrs = model.run_internal_adapt(
                noisy,cfg.sigma,flows=flows,
                ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                nsteps=cfg.internal_adapt_nsteps,
                nepochs=cfg.internal_adapt_nepochs,
                sample_mtype=cfg.adapt_mtype,
                clean_gt = clean,
                region_gt = [2,4,128,256,256,384]
            )
        timer.stop("adapt")

        # -- denoise --
        batch_size = 390*100
        timer.start("deno")
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        if cfg.model_name == "original":
            with th.no_grad():
                noisy_in = rearrange(noisy,'t c h w -> 1 c t h w')
                deno = model.test_forward(noisy_in/imax,flows)
                deno = rearrange(deno,'1 c t h w -> t c h w')
                deno = th.clamp(deno,0,1.)*imax
        else:
            with th.no_grad():
                deno = model.test_forward(noisy/imax,flows)
                deno = th.clamp(deno,0,1.)*imax
        timer.stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)
        print("deno.shape: ",deno.shape)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = pacnet.utils.io.save_burst(deno,out_dir,"deno")

        # -- psnr --
        noisy_psnrs = pacnet.utils.metrics.compute_psnrs(noisy,clean,div=imax)
        psnrs = pacnet.utils.metrics.compute_psnrs(deno,clean,div=imax)
        ssims = pacnet.utils.metrics.compute_ssims(deno,clean,div=imax)
        print(noisy_psnrs)
        print(psnrs)

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.adapt_psnrs.append(adapt_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)

    return results

def load_trained_state(model,use_train):

    # -- skip if needed --
    if not(use_train == "true"): return

    if ca_fwd == "dnls_k":
        # model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=99.ckpt"
        # model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=81-val_loss=1.24e-03.ckpt"
        model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=38-val_loss=1.15e-03.ckpt"
        # model_path = "output/checkpoints/2539a251-8233-49a8-bb4f-db68e8c96559-epoch=26.ckpt"
    elif ca_fwd == "default":
        model_path = "output/checkpoints/dec78611-36a7-4a9e-8420-4e60fe8ea358-epoch=91-val_loss=6.63e-04.ckpt"
    else:
        raise ValueError(f"Uknown ca_fwd [{ca_fwd}]")

    # -- load model state --
    state = th.load(model_path)['state_dict']
    lightning.remove_lightning_load_state(state)
    model.model.load_state_dict(state)
    return model

def save_path_from_cfg(cfg):
    path = Path(cfg.dname) / cfg.vid_name
    train_str = "train" if  cfg.train == "true" else "notrain"
    path = path / "%s_%s" % (cfg.ca_fwd,train_str)

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    cfg.isize = "128_128"
    # cfg.isize = "none"
    cfg.bw = False
    cfg.nframes = 5
    cfg.frame_start = 10
    cfg.frame_end = cfg.frame_start+cfg.nframes-1

    # -- get mesh --
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [0]
    ws,wt,k,bs,stride = [15],[0],[7],[1024*128],[5]
    dnames,sigmas,use_train = ["set8"],[50.],["false"]
    vid_names = ["tractor"]
    # vid_names = ["sunflower","hypersmooth","tractor"]
    # vid_names = ["snowboard","sunflower","tractor","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    flow,isizes,adapt_mtypes = ["true"],["none"],["rand"]
    model_names = ["proposed"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "isize":isizes,"use_train":use_train,"stride":stride,
                 "ws":ws,"wt":wt,"k":k, "bs":bs, "model_name":model_names}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two


    # -- original w/out training --
    exp_lists['model_name'] = ["original"]
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exp_lists['ca_fwd'] = ["default"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a + exps_b
    # exps = exps_b

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
        if exp.model_name == "refactored":
            cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    # print(records)
    # print(records.filter(like="timer"))

    # -- viz report --
    for use_train,tdf in records.groupby("use_train"):
        for ca_group,gdf in tdf.groupby("model_name"):
            for use_flow,fdf in gdf.groupby("flow"):
                agg_psnrs,agg_ssims,agg_dtime = [],[],[]
                agg_mem_res,agg_mem_alloc = [],[]
                print("--- %s (%s,%s) ---" % (ca_group,use_train,use_flow))
                for vname,vdf in fdf.groupby("vid_name"):
                    psnrs = np.stack(vdf['psnrs'])
                    dtime = np.stack(vdf['timer_deno'])
                    mem_res = np.stack(vdf['mem_res'])
                    mem_alloc = np.stack(vdf['mem_alloc'])
                    ssims = np.stack(vdf['ssims'])
                    psnr_mean = psnrs.mean().item()
                    ssim_mean = ssims.mean().item()
                    uuid = vdf['uuid'].iloc[0]

                    # print(dtime,mem_gb)
                    # print(vname,psnr_mean,ssim_mean,uuid)
                    print("%13s: %2.3f %1.3f %s" % (vname,psnr_mean,ssim_mean,uuid))
                    agg_psnrs.append(psnr_mean)
                    agg_ssims.append(ssim_mean)
                    agg_mem_res.append(mem_res.mean().item())
                    agg_mem_alloc.append(mem_alloc.mean().item())
                    agg_dtime.append(dtime.mean().item())
                psnr_mean = np.mean(agg_psnrs)
                ssim_mean = np.mean(agg_ssims)
                dtime_mean = np.mean(agg_dtime)
                mem_res_mean = np.mean(agg_mem_res)
                mem_alloc_mean = np.mean(agg_mem_alloc)
                uuid = tdf['uuid']
                params = ("Ave",psnr_mean,ssim_mean,dtime_mean,
                          mem_res_mean,mem_alloc_mean)
                print("%13s: %2.3f %1.3f %2.3f %2.3f %2.3f" % params)


if __name__ == "__main__":
    main()
