
import numpy as np
import torch as th
from einops import rearrange
import torch.nn.functional as nn_func
from .auxiliary_functions import reflection_pad_3d

def run(vid,nsearch=15,ps_s=15,ps_f=7):
    """

    Add padding and include sliding temporal window for nn search.

    """
    pad = (nsearch-1)//2
    pad_r = ps_f//2 + ps_f

    t,c,h,w = vid.shape
    vid = rearrange(vid,'t c h w -> 1 c t h w')
    device = vid.device
    reflect_pad = (0, pad_r, pad_r)
    const_pad = (pad, pad, pad, pad, 3, 3)

    with th.no_grad():
        min_d,min_i = [],[]
        vid_pad = reflection_pad_3d(vid, reflect_pad)
        vid_pad = nn_func.pad(vid_pad, const_pad,
                              mode='constant', value=-1.)
        for t in range(vid_pad.shape[-3] - 6):
            # -- exec nn --
            sliding_window = vid_pad[..., t:(t + 7), :, :].to(device)
            min_d_i,min_i_i = find_nn(sliding_window,nsearch,ps_s)

            # -- sort --
            min_d_i, sort_i = th.sort(min_d_i, -1)
            min_i_i = min_i_i.gather(-1, sort_i)

            # -- append --
            min_d.append(min_d_i[0,0,0])
            min_i.append(min_i_i[0,0,0])
        min_d = th.stack(min_d)
        min_i = th.stack(min_i)

    return min_d,min_i

def find_nn(seq_pad,nsearch,ps_s):

    # nsearch = 75
    search_max = nsearch-1
    smax = search_max
    pads = (nsearch-1)//2 # 37
    search_mid = pads
    smid = search_mid
    neigh_num = 14
    sub_pad = nsearch + (neigh_num - 1)
    psm1 = ps_s-1
    # print(ps_s,nsearch,psm1)

    # print("seq_pad.shape: ",seq_pad.shape)
    seq_n = seq_pad[:, :, 3:-3, pads:-pads, pads:-pads]
    # print("[original/find_nn] seq_n.shape: ",seq_n.shape)
    b, c, f, h, w = seq_pad.shape
    min_d = th.full((b, 1, f - 6, h - sub_pad, w - sub_pad, 14), float('inf'),
                       dtype=seq_pad.dtype, device=seq_pad.device)
    min_i = th.full(min_d.shape, -(seq_n.numel() + 1),
                       dtype=th.long, device=seq_pad.device)
    i_arange_patch_pad = th.arange(b * f * (h - psm1) * (w - psm1), dtype=th.long,
                                   device=seq_pad.device).view(
                                       b, 1, f, (h - psm1), (w - psm1))
    i_arange_patch_pad = i_arange_patch_pad[..., 3:-3, smid:-smid, smid:-smid]
    i_arange_patch = th.arange(np.array(min_d.shape[0:-1]).prod(), dtype=th.long,
                                  device=seq_pad.device).view(min_d.shape[0:-1])
    for t_s in range(7):
        t_e = t_s - 6 if t_s != 6 else None
        for v_s in range(nsearch):
            v_e = v_s - smax if v_s != smax else None
            for h_s in range(nsearch):
                if h_s == smid and v_s == smid and t_s == 3:
                    continue
                h_e = h_s - smax if h_s != smax else None

                seq_d = ((seq_pad[..., t_s:t_e, v_s:v_e, h_s:h_e] - \
                          seq_n) ** 2).mean(dim=1, keepdim=True)
                seq_d = th.cumsum(seq_d, dim=-1)
                tmp = seq_d[..., 0:-ps_s]
                seq_d = seq_d[..., psm1:]
                # print("seq_d.shape: ",seq_d.shape,tmp.shape)
                seq_d[..., 1:] = seq_d[..., 1:] - tmp

                seq_d = th.cumsum(seq_d, dim=-2)
                tmp = seq_d[..., 0:-ps_s, :]
                seq_d = seq_d[..., psm1:, :]
                # print("seq_d.shape: ",seq_d.shape,tmp.shape)
                seq_d[..., 1:, :] = seq_d[..., 1:, :] - tmp
                # print("seq_d.shape: ",seq_d.shape,tmp.shape)

                neigh_d_max, neigh_i_rel = min_d.max(-1)
                neigh_i_abs = neigh_i_rel + i_arange_patch * psm1

                tmp_i = i_arange_patch_pad + ((t_s - 3) * (h - psm1) * (w - psm1) + \
                                              (v_s - smid) * (w - psm1) + h_s - smid)

                i_change = seq_d < neigh_d_max
                min_d.flatten()[neigh_i_abs[i_change]] = seq_d.flatten()[
                    i_arange_patch[i_change]]
                min_i.flatten()[neigh_i_abs[i_change]] = tmp_i[i_change]

    return min_d, min_i
