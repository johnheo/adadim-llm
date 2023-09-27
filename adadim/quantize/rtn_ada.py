import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict
from transformers.models.llama.modeling_llama import LlamaForCausalLM

__all__ = ["run_rtn_ada"]

def get_named_linears(module, cls_name=nn.Linear):
    return {name: m for name, m in module.named_modules() if isinstance(m, cls_name)}

def get_blocks(model):
    if isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers
    
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))

@torch.inference_mode()
def auto_dim_block(module, module_kwargs,
                     w_bit, q_config,
                     input_feat,
                     adaptive):
    from .quantizer import pseudo_quantize_tensor
    # firstly, get the weight quantize function
    if w_bit is not None:
        def w_quantize_func(p, dim): return pseudo_quantize_tensor(
            p, n_bit=w_bit, per_ic=dim=='ic', **q_config,
        ).detach()
    else:
        def w_quantize_func(p): return p

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    named_linears = get_named_linears(module)
    best_dims = defaultdict()
    for n, m in named_linears.items():
        # w: co, ci
        # x: n, ci
        kwargs={}
        block=None
        if 'self_attn' in n:
            if 'o_' in n:
                x = input_feat['self_attn.o_proj']
            else: # qkv
                block=module.self_attn
                x = input_feat['self_attn.q_proj']
                kwargs=module_kwargs
        else:
            if 'down_' in n:
                x = input_feat['mlp.down_proj']
            else: # up, gate
                block=module.mlp
                x = input_feat['mlp.gate_proj']
        if block is None:
            block = m
        
        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        best_error = float('inf')
        best_dim = None
        sd = {}
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        sd['org'] = org_sd
        candid = ['oc', 'ic'] if adaptive else ['oc']
        for dim in candid:
            m.weight.data = w_quantize_func(m.weight.data, dim)
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_dim = dim
            sd[dim] = {k: v.cpu() for k, v in block.state_dict().items()}
            block.load_state_dict(org_sd)
        block.load_state_dict(sd[best_dim])
        print(f" * auto dim | {n}: best={best_dim}")
        del sd

@torch.inference_mode()
def run_rtn_ada(
    model, enc,
    w_bit, q_config,
    n_samples=512, seqlen=512,
    calib_data="pileval",
    adaptive=False,
):
    from ..utils.calib_data import get_calib_dataset
    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)

    print('samples shape', samples.shape)
    print('adaptive? ', adaptive)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    
    gc.collect()
    torch.cuda.empty_cache()

    dims = []
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running RTN-ada..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        # Clear GPU memory
        torch.cuda.empty_cache()
        auto_dim_block(
            layer, layer_kwargs,
            w_bit=w_bit, q_config=q_config,
            input_feat=input_feat,
            adaptive=adaptive,
        )
        layer = layer.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        
    return model
