

import numpy as np
import torch as th
from einops import rearrange
import torch.nn.functional as nn_func
from .auxiliary_functions import reflection_pad_3d
from torch.nn.functional import unfold,fold

def run(vid, min_i,nsearch=15,ps=15,ps_f=7):
    """

    Add padding and include sliding temporal window for nn search.

    """

    t,c,h,w = vid.shape
    vid = rearrange(vid,'t c h w -> 1 c t h w')
    device = vid.device
    pad_r = (ps_f//2) + ps_f
    print("pad_r: ",pad_r)
    reflect_pad = (0, pad_r, pad_r)
    pad = (nsearch-1)//2
    print("pad: ",pad)
    const_pad = (pad, pad, pad, pad, 3, 3)
    with th.no_grad():
        vid_pad = reflection_pad_3d(vid, reflect_pad)
        vid_pad = nn_func.pad(vid_pad, const_pad,
                              mode='constant', value=-1.)
        print("[og pad]: ",vid_pad.shape)
        sim_layers = []
        for t in range(vid_pad.shape[-3] - 6):
            # -- exec nn --
            sliding_window = vid_pad[..., t:(t + 7), :, :].to(device)
            sliding_window = sliding_window[..., 4:-4, 4:-4]
            min_i_t = min_i[[t],None,None]
            sim_layer_t = create_layers(sliding_window,min_i_t,nsearch,ps_f)
            # sim_layer_t = create_layers_og(sliding_window,min_i_t)

            # -- append --
            sim_layers.append(sim_layer_t)
        sim_layers = th.stack(sim_layers)
    return sim_layers

def create_layers(seq_pad, min_i, nsearch, ps):

    smid = nsearch//2
    pad = 2*(ps//2) # 6 = 2 * 3 = 2 * (7//2)
    rm_pix = 2*(ps//2)
    print("seq_pad.shape: ",seq_pad.shape)

    b, c, f, h, w = seq_pad.shape
    pdim = ps*ps*c
    layer_h = h - (nsearch - 1)
    layer_w = w - (nsearch - 1)
    in_layers = th.full((b, 15, pdim, f - rm_pix, layer_h, layer_w),
                        float('nan'), dtype=seq_pad.dtype, device=seq_pad.device)

    self_i = th.arange(b * f * (h - rm_pix) * (w - rm_pix), dtype=th.long,
                       device=seq_pad.device).view(b, 1, f, h - rm_pix, w - rm_pix)
    self_i = self_i[..., 3:-3, smid:-smid, smid:-smid].unsqueeze(-1)
    # print("self_i.shape: ",self_i.shape)
    # print("min_i.shape: ",min_i.shape)
    min_i = th.cat((self_i, min_i), dim=-1)

    f_ind = 0
    inds_h = h - (nsearch - 1 + pad) # h - 80 = h - (nsearch - 1 + pad) = h - (75-1+6)
    inds_w = w - (nsearch - 1 + pad)
    min_i = min_i.permute(0, 2, 5, 1, 3, 4)
    min_i = min_i.reshape(b * (f - rm_pix) * 15, inds_h, inds_w)
    # print("in_layers.shape: ",in_layers.shape)

    for map_v_s in range(ps):
        for map_h_s in range(ps):
            edge_v = h - (nsearch-1) - map_v_s
            edge_h = w - (nsearch-1) - map_h_s
            # (h - 80 - (h - ps4 - map_v_s) % ps)
            end_v = (inds_h - edge_v % ps)
            end_h = (inds_w - edge_h % ps)
            min_i_tmp = min_i[..., map_v_s:end_v:ps, \
                                   map_h_s:end_h:ps].flatten()
            # h - 80 - (h - 74 - map_v_s) % 7
            # print(map_v_s,end_v,h-(nsearch-1),h,nsearch)
            # print(map_v_s,end_v)
            # print(map_h_s,end_h)
            # print("min_i_tmp.shape: ",min_i_tmp.shape)
            layers_pad_tmp = unfold(seq_pad.transpose(1, 2).
                                    reshape(b * f, 3, h, w), (ps, ps))
            layers_pad_tmp = layers_pad_tmp.transpose(0, 1).reshape(pdim, -1)
            layers_pad_tmp = layers_pad_tmp[:, min_i_tmp]
            layers_pad_tmp = layers_pad_tmp.view(pdim, b * (f - 6) * 15,
                                                 (edge_v // ps) * (edge_h // ps))
            layers_pad_tmp = layers_pad_tmp.transpose(0, 1)
            layers_pad_tmp = fold(
                input=layers_pad_tmp,
                output_size=(edge_v - edge_v % ps, edge_h - edge_h % ps),
                kernel_size=(ps, ps), stride=ps)
            layers_pad_tmp = layers_pad_tmp.view(b, f - 6, 15, 3, \
                edge_v - edge_v % ps, edge_h - edge_h % ps)
            layers_pad_tmp = layers_pad_tmp.permute(0, 2, 3, 1, 4, 5)
            # print("layers_pad_tmp.shape: ",layers_pad_tmp.shape)
            # print("info: ",map_v_s,(h - (nsearch-1) - edge_v % ps))

            in_layers[:, :, f_ind:f_ind + 3, :, \
                      map_v_s:(h - (nsearch-1) - edge_v % ps),
                      map_h_s:(w - (nsearch-1) - edge_h % ps)] = layers_pad_tmp
            f_ind = f_ind + 3
    in_layers = in_layers[..., 6:-6, 6:-6]

    return in_layers

def create_layers_og(seq_pad, min_i):

    b, c, f, h, w = seq_pad.shape
    in_layers = th.full((b, 15, 147, f - 6, h - 74, w - 74),
                           float('nan'), dtype=seq_pad.dtype, device=seq_pad.device)

    self_i = th.arange(b * f * (h - 6) * (w - 6), dtype=th.long,
                          device=seq_pad.device).view(b, 1, f, h - 6, w - 6)
    self_i = self_i[..., 3:-3, 37:-37, 37:-37].unsqueeze(-1)
    min_i = th.cat((self_i, min_i), dim=-1)

    f_ind = 0
    min_i = min_i.permute(0, 2, 5, 1, 3, 4)
    min_i = min_i.reshape(b * (f - 6) * 15, h - 80, w - 80)

    # print("in_layers.shape: ",in_layers.shape)
    for map_v_s in range(7):
        for map_h_s in range(7):
            # print(map_v_s,(h - 80 - (h - 74 - map_v_s) % 7))
            # print(map_h_s,(w - 80 - (w - 74 - map_h_s) % 7))
            min_i_tmp = min_i[..., map_v_s:(h - 80 - (h - 74 - map_v_s) % 7):7, \
                                   map_h_s:(w - 80 - (w - 74 - map_h_s) % 7):7].flatten()
            layers_pad_tmp = unfold(seq_pad.transpose(1, 2).
                                    reshape(b * f, 3, h, w), (7, 7))
            layers_pad_tmp = layers_pad_tmp.transpose(0, 1).reshape(147, -1)
            layers_pad_tmp = layers_pad_tmp[:, min_i_tmp]
            layers_pad_tmp = layers_pad_tmp.view(147, b * (f - 6) * 15,
                                                ((h - 74 - map_v_s) // 7) * ((w - 74 - map_h_s) // 7))
            layers_pad_tmp = layers_pad_tmp.transpose(0, 1)
            layers_pad_tmp = fold(
                input=layers_pad_tmp,
                output_size=(h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                             w - 74 - map_h_s - (w - 74 - map_h_s) % 7),
                kernel_size=(7, 7), stride=7)
            layers_pad_tmp = layers_pad_tmp.view(b, f - 6, 15, 3, \
                h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                w - 74 - map_h_s - (w - 74 - map_h_s) % 7)
            layers_pad_tmp = layers_pad_tmp.permute(0, 2, 3, 1, 4, 5)
            # print("layers_pad_tmp.shape: ",layers_pad_tmp.shape)

            in_layers[:, :, f_ind:f_ind + 3, :, \
                map_v_s:(h - 74 - (h - 74 - map_v_s) % 7), 
                map_h_s:(w - 74 - (w - 74 - map_h_s) % 7)] = layers_pad_tmp
            f_ind = f_ind + 3
    in_layers = in_layers[..., 6:-6, 6:-6]

    return in_layers
