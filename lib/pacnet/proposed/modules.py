import torch.nn as nn
import torch
import torch as th
from torch.nn.functional import unfold
from torch.nn.functional import fold
import torch.nn.functional as nn_func
from einops import rearrange

import numpy as np
import math

from .auxiliary_functions import *
from .find_nn import run as find_nn
from .find_nn_multigpu import run as find_nn_multigpu
from .create_layers import run as create_layers


class Logger(object):
    def __init__(self, fname="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class SepConvFM2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvFM2D, self).__init__()

        if 0 == l_ind:
            self.f_in = 150
        else:
            self.f_in = max(math.ceil(150 / (2 ** l_ind)), 150)

        self.f_out = max(math.ceil(self.f_in / 2), 150)
        self.n_in = math.ceil(15 / (2 ** l_ind))
        self.n_out = math.ceil(self.n_in / 2)
        self.vh_groups = (self.f_in // 3) * self.n_in
        self.f_groups = self.n_in
        self.n_groups = self.f_out
        self.f_in_g = self.f_in * self.n_in
        self.f_out_g = self.f_out * self.n_in
        self.n_in_g = self.n_in * self.n_groups
        self.n_out_g = self.n_out * self.n_groups
        self.conv_vh = nn.Conv2d(in_channels=self.f_in_g, out_channels=self.f_in_g,
                                 kernel_size=(7, 7), bias=False, groups=self.vh_groups)
        self.conv_f = nn.Conv2d(in_channels=self.f_in_g, out_channels=self.f_out_g,
                                kernel_size=(1, 1), bias=False, groups=self.f_groups)
        self.conv_n = nn.Conv2d(in_channels=self.n_in_g, out_channels=self.n_out_g,
                                kernel_size=(1, 1), bias=False, groups=self.n_groups)

    def forward(self, x):
        b, n, f, v, h = x.shape  # batches, neighbors, features, horizontal, vertical

        x = reflection_pad_2d(x.reshape(b, n * f, v, h), (3, 3))
        x = self.conv_vh(x)
        x = self.conv_f(x).reshape(b, n, self.f_out, v, h)
        x = self.conv_n(x.transpose(1, 2).reshape(b, self.f_out * n, v, h)).\
            reshape(b, self.f_out, self.n_out, v, h).transpose(1, 2)

        return x


class SepConvOut2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvOut2D, self).__init__()

        self.f_in = max(math.ceil(150 / (2 ** l_ind)), 150)
        self.vh_groups = self.f_in // 3
        self.conv_vh = nn.Conv2d(in_channels=self.f_in, out_channels=self.f_in,
                                 kernel_size=(7, 7), bias=False, groups=self.vh_groups)
        self.conv_f = nn.Conv2d(in_channels=self.f_in, out_channels=3,
                                kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x = reflection_pad_2d(x.squeeze(1), (3, 3))
        x = self.conv_vh(x)
        x = self.conv_f(x)

        return x


class SepConvReF2D(nn.Module):
    def __init__(self):
        super(SepConvReF2D, self).__init__()

        self.sep_conv = SepConvFM2D(0)
        self.b = nn.Parameter(torch.zeros((1, self.sep_conv.n_out, self.sep_conv.f_out, 1, 1), dtype=torch.float32))
        self.re = nn.ReLU()

    def forward(self, x):
        x = self.sep_conv(x)
        x = x + self.b
        x = self.re(x)

        return x

    def extra_repr(self):
        return 'b.shape=' + str(tuple(self.b.shape))


class SepConvBnReM2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvBnReM2D, self).__init__()

        self.sep_conv = SepConvFM2D(l_ind)
        self.bn = nn.BatchNorm2d(num_features=self.sep_conv.f_out * self.sep_conv.n_out)
        self.re = nn.ReLU()

    def forward(self, x):
        x = self.sep_conv(x)
        b, n, f, v, h = x.shape  # batches, neighbors, features, horizontal, vertical
        x = self.bn(x.reshape(b, n * f, v, h)).reshape(b, n, f, v, h)
        x = self.re(x)

        return x


class SepConvOutB2D(nn.Module):
    def __init__(self, l_ind):
        super(SepConvOutB2D, self).__init__()

        self.sep_conv = SepConvOut2D(l_ind)
        self.b = nn.Parameter(torch.zeros((1, 3, 1, 1), dtype=torch.float32))

    def forward(self, x):
        x = self.sep_conv(x)
        x = x + self.b

        return x

    def extra_repr(self):
        return 'b.shape=' + str(tuple(self.b.shape))


class SepConvNet2D(nn.Module):
    def __init__(self):
        super(SepConvNet2D, self).__init__()

        self.sep_conv_block0 = SepConvReF2D()
        for i in range(1, 4):
            self.add_module('sep_conv_block{}'.format(i), SepConvBnReM2D(i))
        self.add_module('sep_conv_block{}'.format(4), SepConvOutB2D(4))

    def forward(self, x_f):
        for name, layer in self.named_children():
            x_f = layer(x_f)

        return x_f


class ResNn(nn.Module):
    def __init__(self):
        super(ResNn, self).__init__()

        self.sep_conv_net = SepConvNet2D()


    def forward(self, x_f, x, x_valid):

        x_f = self.sep_conv_net(x_f).squeeze(2)

        return x_valid - x_f


class PaCNet(nn.Module):

    def __init__(self,ws=25,wt=5,k=15,ps_f=7,ps_s=15,ngpus=1,
                 nn_bs_list=None,time_chunk=-1):
        super(PaCNet, self).__init__()
        self.vid_cnn = VidCnn(ws,wt,k,ps_f,ps_s,ngpus,nn_bs_list,time_chunk)
        self.tf_net = TfNet()

    def forward(self,vid,flows,clean=None):
        pass

    def test_forward(self,vid,flows,clean=None):
        device = vid.device
        deno = self.vid_cnn.test_forward(vid,flows,clean)
        th.cuda.empty_cache()
        deno = self.tf_net.test_forward(vid,deno,device)
        return deno

class VidCnn(nn.Module):
    def __init__(self,ws=25,wt=5,k=15,ps_f=7,ps_s=15,ngpus=1,
                 nn_bs_list=None,time_chunk=-1):
        super(VidCnn, self).__init__()
        self.ws = ws
        self.wt = wt
        self.k = k
        self.ps_s = ps_s
        self.ps_f = ps_f
        self.ngpus = ngpus
        self.nn_bs_list = nn_bs_list
        self.time_chunk = time_chunk
        self.res_nn = ResNn()

    def pad_vid(self,vid):
        pad_r = self.ps_f//2 + self.ps_f
        pad_c = self.ws//2

        vid_pad = nn_func.pad(vid, [pad_r,]*4,
                              mode='reflect')
        vid_pad = nn_func.pad(vid_pad, [pad_c,]*4,
                              mode='constant', value=-1)
        return vid_pad

    def compute_sims(self, noisy_vid, clean_vid, flows):
        # -- unpack --
        nframes,c,h,w = noisy_vid.shape
        device = noisy_vid.device

        # -- shape to expected for original --
        with torch.no_grad():
            # -- add padding --
            noisy_pad = self.pad_vid(noisy_vid)
            clean_pad = self.pad_vid(clean_vid)

            # -- temporal crop for pacnet processing --
            inds = self.find_sorted_nn(noisy_pad,flows).to(device)

            # -- non-local search --
            clean_pad = clean_pad[..., 4:-4, 4:-4].contiguous()

            # -- sim video --
            sim_vids = create_layers(clean_pad, inds, self.ws,
                                     self.ps_s, self.ps_f, add_padding=False)
        return sim_vids

    def test_forward(self, noisy_vid, flows):
        nframes,c,h,w = noisy_vid.shape
        device = noisy_vid.device
        pad_r = self.ps_f//2 + self.ps_f
        pad_c = self.ws//2
        reflect_pad = (0, pad_r, pad_r)
        const_pad = (pad_c, pad_c, pad_c, pad_c)#, 3, 3)
        time_chunk = nframes if self.time_chunk > 0 else self.time_chunk
        time_chunk = min(nframes,time_chunk)
        noisy_vid = noisy_vid.cpu()
        with torch.no_grad():

            # -- shell to fill --
            deno = th.zeros_like(noisy_vid)

            # -- add padding --
            noisy_vid_pad = nn_func.pad(noisy_vid, [pad_r,]*4,
                                        mode='reflect')
            noisy_vid_pad = nn_func.pad(noisy_vid_pad, const_pad,
                                        mode='constant', value=-1)
            noisy_vid_pad = noisy_vid_pad#.to(device)

            # -- process each frame --
            for ti in range(nframes):
                # -- cleanup --
                # print(th.cuda.current_device())
                th.cuda.empty_cache()

                # -- temporal crop for pacnet processing --
                t_start = max(ti - self.time_chunk//2,0)
                t_end = min(t_start + self.time_chunk,nframes)
                t_start = max(t_end - self.time_chunk,0)
                tj = np.argmin(np.abs(np.arange(t_start,t_end) - ti))
                # print(ti,tj,t_start,t_end)
                noisy_search = noisy_vid_pad[t_start:t_end].to(device)

                # -- non-local search --
                flows_i = self.tslice_flows(flows,t_start,t_end)
                inds = self.find_sorted_nn(noisy_search,flows_i,tj,tj+1)
                noisy_search = noisy_search[..., 4:-4, 4:-4].contiguous()
                inds_tj = inds.to(device)

                # -- creating sim layer --
                # inds_tj = inds[[tj]].to(device)
                # print("inds.shape: ",inds.shape)
                sim_vid = create_layers(noisy_search, inds_tj, self.ws,
                                        self.ps_s, self.ps_f, add_padding=False)
                sim_vid = rearrange(sim_vid,'k t c h w -> t k c h w')

                # -- get middle img --
                cc = self.ws//2 + (self.ps_f-1)
                noisy_chunk = noisy_search[...,cc:-cc,cc:-cc]
                noisy_ti = noisy_search[[tj],...,cc:-cc,cc:-cc]

                # -- exec vid cnn --
                # print(noisy_ti.shape,noisy_chunk.shape,sim_vid.shape)
                # print(noisy_ti.device,noisy_chunk.device,sim_vid.device)
                deno_ti = self(noisy_ti,noisy_chunk,sim_vid)
                deno[ti,...] = deno_ti[...].clamp(min=0, max=1).cpu()
                del sim_vid,inds

        th.cuda.synchronize()
        # print("done.")
        return deno.to(device)

    def tslice_flows(self,flows,ts,te):
        if flows is None: return None
        flows_s = {}
        for aflow in flows:
            flows_s[aflow] = flows[aflow][ts:te]
        return flows_s

    def forward(self, seq_valid, seq_valid_full, in_layers):
        # -- patch weights --
        in_weights = (in_layers - in_layers[:,0:1, ...]) ** 2
        b, n, f, h, w = in_weights.shape
        in_weights = in_weights.view(b, n, f // 3, 3, h, w).mean(2)
        in_layers = torch.cat((in_layers, in_weights), 2)
        # print("in_layers.shape: ",in_layers.shape)
        deno = self.res_nn(in_layers, seq_valid_full, seq_valid)
        return deno

    def find_sorted_nn(self, seq_in, flows=None, tstart=None, tend=None):
        if self.ngpus > 1:
            dists, inds = find_nn_multigpu(seq_in,flows,ps=self.ps_s,k=self.k,
                                           ws=self.ws,wt=self.wt,ps_f=self.ps_f,
                                           reshape_output=True,add_padding=False,
                                           bs_list=self.nn_bs_list,
                                           tstart=tstart,tend=tend)
        else:
            dists, inds = find_nn(seq_in,flows,ps=self.ps_s,k=self.k,
                                  ws=self.ws,wt=self.wt,ps_f=self.ps_f,
                                  reshape_output=True,add_padding=False,
                                  bs=self.nn_bs_list[0],
                                  tstart=tstart,tend=tend)
        return inds.cpu()


class ImCnn(nn.Module):
    def __init__(self):
        super(ImCnn, self).__init__()

        self.res_nn = ResNn()

    def find_nn(self, seq_pad):
        seq_n = seq_pad[..., 37:-37, 37:-37]
        b, c, f, h, w = seq_pad.shape
        min_d = torch.full((b, 1, f, h - 80, w - 80, 14), float('inf'),
                           dtype=seq_pad.dtype, device=seq_pad.device)
        min_i = torch.full(min_d.shape, -(seq_n.numel() + 1),
                           dtype=torch.long, device=seq_pad.device)
        i_arange_patch_pad = torch.arange(b * f * (h - 6) * (w - 6), dtype=torch.long,
                                          device=seq_pad.device).view(b, 1, f, h - 6, w - 6)
        i_arange_patch_pad = i_arange_patch_pad[..., 37:-37, 37:-37]
        i_arange_patch = torch.arange(np.array(min_d.shape[0:-1]).prod(), dtype=torch.long,
                                      device=seq_pad.device).view(min_d.shape[0:-1])

        for v_s in range(75):
            v_e = v_s - 74 if v_s != 74 else None
            for h_s in range(75):
                if h_s == 37 and v_s == 37:
                    continue
                h_e = h_s - 74 if h_s != 74 else None

                seq_d = ((seq_pad[..., v_s:v_e, h_s:h_e] - seq_n) ** 2).mean(dim=1, keepdim=True)

                seq_d = torch.cumsum(seq_d, dim=-1)
                tmp = seq_d[..., 0:-7]
                seq_d = seq_d[..., 6:]
                seq_d[..., 1:] = seq_d[..., 1:] - tmp

                seq_d = torch.cumsum(seq_d, dim=-2)
                tmp = seq_d[..., 0:-7, :]
                seq_d = seq_d[..., 6:, :]
                seq_d[..., 1:, :] = seq_d[..., 1:, :] - tmp

                neigh_d_max, neigh_i_rel = min_d.max(-1)
                neigh_i_abs = neigh_i_rel + i_arange_patch * 14
                tmp_i = i_arange_patch_pad + ((v_s - 37) * (w - 6) + h_s - 37)

                i_change = seq_d < neigh_d_max
                min_d.flatten()[neigh_i_abs[i_change]] = seq_d.flatten()[i_arange_patch[i_change]]
                min_i.flatten()[neigh_i_abs[i_change]] = tmp_i[i_change]

        return min_d, min_i

    def create_layers(self, seq_pad, min_i):

        b, c, f, h, w = seq_pad.shape

        in_layers = torch.full((b, 15, 147, f, h - 74, w - 74),
                               float('nan'), dtype=seq_pad.dtype, device=seq_pad.device)

        self_i = torch.arange(b * f * (h - 6) * (w - 6), dtype=torch.long,
                              device=seq_pad.device).view(b, 1, f, h - 6, w - 6)
        self_i = self_i[..., 37:-37, 37:-37].unsqueeze(-1)
        min_i = torch.cat((self_i, min_i), dim=-1)

        f_ind = 0
        min_i = min_i.permute(0, 2, 5, 1, 3, 4)
        min_i = min_i.reshape(b * f * 15, h - 80, w - 80)

        for map_v_s in range(7):
            for map_h_s in range(7):
                min_i_tmp = min_i[..., map_v_s:(h - 80 - (h - 74 - map_v_s) % 7):7, \
                                       map_h_s:(w - 80 - (w - 74 - map_h_s) % 7):7].flatten()

                layers_pad_tmp = unfold(seq_pad.transpose(1, 2).
                                        reshape(b * f, 3, h, w), (7, 7))
                layers_pad_tmp = layers_pad_tmp.transpose(0, 1).reshape(147, -1)
                layers_pad_tmp = layers_pad_tmp[:, min_i_tmp]
                layers_pad_tmp = layers_pad_tmp.view(147, b * f * 15,
                                                    ((h - 74 - map_v_s) // 7) * ((w - 74 - map_h_s) // 7))
                layers_pad_tmp = layers_pad_tmp.transpose(0, 1)
                layers_pad_tmp = fold(
                    input=layers_pad_tmp,
                    output_size=(h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                                 w - 74 - map_h_s - (w - 74 - map_h_s) % 7),
                    kernel_size=(7, 7), stride=7)
                layers_pad_tmp = layers_pad_tmp.view(b, f, 15, 3, \
                    h - 74 - map_v_s - (h - 74 - map_v_s) % 7, 
                    w - 74 - map_h_s - (w - 74 - map_h_s) % 7)
                layers_pad_tmp = layers_pad_tmp.permute(0, 2, 3, 1, 4, 5)

                in_layers[:, :, f_ind:f_ind + 3, :, \
                    map_v_s:(h - 74 - (h - 74 - map_v_s) % 7), 
                    map_h_s:(w - 74 - (w - 74 - map_h_s) % 7)] = layers_pad_tmp
                f_ind = f_ind + 3
        in_layers = in_layers[..., 6:-6, 6:-6]

        return in_layers

    def forward(self, seq_in, gpu_usage):
        if gpu_usage == 1 and torch.cuda.is_available():
            min_i = self.find_sorted_nn(seq_in.cuda()).cpu()
        else:
            min_i = self.find_sorted_nn(seq_in)

        in_layers = self.create_layers(seq_in, min_i)
        in_layers = in_layers.squeeze(-3)

        seq_valid_full = seq_in[..., 43:-43, 43:-43]
        seq_valid = seq_valid_full[..., 0, :, :]
        seq_valid_full = seq_valid_full.squeeze(-3)

        in_weights = (in_layers - in_layers[:, 0:1, ...]) ** 2
        b, n, f, v, h = in_weights.shape
        in_weights = in_weights.view(b, n, f // 3, 3, v, h).mean(2)
        in_layers = torch.cat((in_layers, in_weights), 2)

        seq_out = self.res_nn(in_layers, seq_valid_full, seq_valid)

        return seq_out

    def find_sorted_nn(self, seq_in):
        seq_pad_nn = seq_in
        min_d, min_i = self.find_nn(seq_pad_nn)
        min_d, sort_i = torch.sort(min_d, -1)
        min_i = min_i.gather(-1, sort_i)

        return min_i


class ConvRe3DF(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvRe3DF, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), bias=True,
                              padding=(0, 1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class ConvBnRe3DM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBnRe3DM, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3, 3), bias=False,
                              padding=(0, 1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class ConvRe2DF(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvRe2DF, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), bias=False,
                              padding=(1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class ConvBnRe2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBnRe2D, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), bias=False,
                              padding=(1, 1), padding_mode='zeros')
        self.re = nn.LeakyReLU()

        self.a = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        x = self.conv(x)
        x = self.re(x)

        return x


class Conv2DL(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv2DL, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3, 3), bias=True,
                              padding=(1, 1), padding_mode='zeros')

    def forward(self, x):
        x = self.conv(x)

        return x


class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()

        self.conv_3d_block0 = ConvRe3DF(in_ch=6, out_ch=48)
        
        for i in range(1, 3):
            out_ch_tmp = 48 if i < 2 else 96
            self.add_module('conv_3d_block{}'.format(i), ConvBnRe3DM(in_ch=48, out_ch=out_ch_tmp))

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class Conv2DNet(nn.Module):
    def __init__(self):
        super(Conv2DNet, self).__init__()

        for i in range(16):
            self.add_module('conv_2d_block{}'.format(i), ConvBnRe2D(in_ch=96, out_ch=96))
        
        self.add_module('conv_2d_block{}'.format(16), Conv2DL(in_ch=96, out_ch=3))

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)

        return x


class TfNet3D(nn.Module):
    def __init__(self):
        super(TfNet3D, self).__init__()

        self.conv_3d_net = Conv3DNet()
        self.conv_2d_net = Conv2DNet()

    def forward(self, x):
        x = self.conv_3d_net(x).squeeze(-3)
        x = self.conv_2d_net(x)

        return x


class TfNet(nn.Module):
    def __init__(self):
        super(TfNet, self).__init__()
        self.conv_net = TfNet3D()

    def test_forward(self, noisy, deno0, device="cuda:0"):
        t,c,h,w,device = *noisy.shape,noisy.device
        deno_vid = th.zeros_like(deno0)
        cat_vid = torch.cat((noisy, deno0), dim=-3)
        cat_vid = proposed_tf_pad(cat_vid).to(device)
        with th.no_grad():
            for t_s in range(cat_vid.shape[0] - 6):
                sliding_window = cat_vid[t_s:(t_s + 7), ...].to(device)
                deno_t = self(sliding_window).clamp(min=0, max=1)
                deno_vid[t_s, ...] = deno_t
        return deno_vid

    def forward(self, x):
        x = rearrange(x,'t c h w -> 1 c t h w')
        xn = x[:, 3:6, 3, :, :].clone()
        x = self.conv_net(x)
        x = xn - x # 1 c h w
        x = rearrange(x,'1 c h w -> 1 c h w')

        return x

def proposed_tf_pad(in_seq, pad=3):
    device = in_seq.device
    in_seq = in_seq.cpu()
    in_seq = rearrange(in_seq,'t c h w -> 1 c t h w')
    out = reflection_pad_t_3d(in_seq, pad)
    out = rearrange(out,'1 c t h w -> t c h w')
    return out.to(device)
