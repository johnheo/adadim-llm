# make quantization more intuitive
import math
import time

import torch
import torch.nn as nn
import transformers

from .quant import quantize, pseudo_quantize_tensor

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows, self.columns = layer.weight.data.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        # inp: b,t,c
        bsz = 1
        inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + bsz)
        self.nsamples += bsz
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp @ inp.t()


    def fasterquant(
        self, blocksize=128, percdamp=.01, w_bit=8, groupsize=-1,
        actorder:str=None,
        static_groups=False,
        cfg='oc', # 'ic' or 'oc',
        grads={}
    ):
        def gptq_update(W, H, cfg, actorder, static_groups=False):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)
            if static_groups:
                assert groupsize != -1
                import copy
                if cfg == 'oc':
                    groups = []
                    for i in range(0, self.columns, groupsize):
                        quantizer = copy.deepcopy(self.quantizer)
                        quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                        groups.append(quantizer)
            if actorder:
                x_scales = torch.diag(H)
                perm = torch.argsort(x_scales, descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]
                invperm = torch.argsort(perm)

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            GRAD = torch.zeros_like(W)
            
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone() # [oc, ic_blk]
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                # local update
                for i in range(count):
                    w = W1[:, i] # [oc]
                    d = Hinv1[i, i]
                    if cfg=='ic':
                        # per-col quant; per-ic quant
                        q = pseudo_quantize_tensor(w, n_bit=w_bit, 
                                                    q_group_size=groupsize,
                                                    vector_quant=True,
                                                    )
                    else:
                        if groupsize != -1:
                            if not static_groups:
                                if (i1 + i) % groupsize == 0:
                                    self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                            else: # statically fetch precomputed quantizers
                                idx = i1 + i
                                if actorder:
                                    idx = perm[idx]
                                self.quantizer = groups[idx // groupsize]

                        q = quantize(w.unsqueeze(1), 
                                     self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                                     ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    # [oc, ic_fp] -= [oc, 1] @ [1, ic_fp]
                    W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                    Err1[:, i] = err1
                    GRAD[:, i1:i2][:, i:] += err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
                
                # global update
                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2
                # [oc, ic_rest] -= [oc, ic_blk] @ [ic_blk, ic_rest]
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
                GRAD[:, i2:] += Err1 @ Hinv[i1:i2, i2:]
            # grads.update({self.name: GRAD})
            torch.cuda.synchronize()
            if actorder:
                Q = Q[:, invperm]

            # return Q, grads
            return Q


        H = self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        if dead.sum().item() > 0:
            print('death exists ', dead.sum().item())

        W = self.layer.weight.detach().clone().float()
        W[:, dead] = 0
        print('* {} | GPTQ{}-{} with w={w_bit} gsize={groupsize} blksize={blocksize} stag={static_groups}'.format(
                                                                self.name,
                                                                f'_R-{actorder}' if actorder else '', 
                                                                cfg,
                                                                w_bit=w_bit,
                                                                groupsize=groupsize,
                                                                blocksize=blocksize,
                                                                static_groups=static_groups,))
        # Q, grads = gptq_update(W, H, cfg, actorder, static_groups)
        Q = gptq_update(W, H, cfg, actorder, static_groups)
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # return grads

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.Losses = None
        torch.cuda.empty_cache()
