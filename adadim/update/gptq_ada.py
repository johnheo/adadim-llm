import torch
import torch.nn as nn
import tqdm
import gc
import functools
from collections import defaultdict

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM


from .gptq import GPTQ
from .quant import Quantizer

__all__ = ["run_gptq_ada"]

def get_named_linears(module, cls_name=nn.Linear):
    return {name: m for name, m in module.named_modules() if isinstance(m, cls_name)}


def get_blocks(model):
    name = model.__class__.__name__.lower()
    if 'qwen' in name:
        layers = model.transformer.h
    elif isinstance(model, LlamaForCausalLM):
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    else:
        raise NotImplementedError(type(model))
    return layers
    
def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(device)
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    else:
        raise NotImplementedError(type(model))


@torch.inference_mode()
def auto_dim_block(module, module_kwargs,
                     w_bit, q_config,
                     input_feat,
                     output_feat,
                     adaptive
                     ):
    from ..quantize.quantizer import pseudo_quantize_tensor
    # firstly, get the weight quantize function
    if w_bit is not None:
        def w_quantize_func(p, dim): return pseudo_quantize_tensor(
            p, n_bit=w_bit, per_ic=dim=='ic', **q_config,
        ).detach()
    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    gptq = {}
    named_linears = get_named_linears(module)
    for n, m in named_linears.items():
        # w: co, ci
        # x: n, ci
        kwargs={}
        block=None
        x = input_feat[n]
        org_out = output_feat[n]
        if 'self_attn' in n:
            if 'o_' not in n: # qkv
                block=module.self_attn
                kwargs=module_kwargs
        else:
           if 'down_' not in n: # up, gate
                block=module.mlp
        
        
        if block is None:
            block = m
            x = x.to(next(block.parameters()).device)
            org_out = org_out.to(next(block.parameters()).device)
        else:
            x = x.to(next(block.parameters()).device)
            with torch.no_grad():
                org_out = block(x, **kwargs)
                if isinstance(org_out, tuple):
                    org_out = org_out[0]

        # config gptq
        gptq[n] = GPTQ(m)
        gptq[n].quantizer = Quantizer()
        gptq[n].name = n
        gptq[n].quantizer.configure(w_bit, perchannel=True, sym=False, mse=False)
        # do add batch op

        org_shape = x.shape
        # NOTE: extend seqlen to 1024
        # x = x.view(x.shape[0]//2, -1, x.shape[-1])
        bsz = x.shape[0]
        for i in range(bsz):
            gptq[n].add_batch(x[i], org_out[i])

        best_error = float('inf')
        best_dim = None
        sd = {}
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        sd['org'] = org_sd

        candidates = ['oc', 'ic'] if adaptive else ['oc']
        for dim in candidates :
            m.weight.data = w_quantize_func(m.weight.data, dim)
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_dim = dim
            block.load_state_dict(org_sd)
        candidates = [str(best_dim)]
        
        for dim in candidates:
            if dim == 'oc':
                gptq[n].fasterquant(
                    percdamp=.01, w_bit=w_bit, groupsize=q_config['q_group_size'], 
                    actorder='sx',
                    static_groups=True,
                    cfg=dim,
                )
            else: # ic
                gptq[n].fasterquant(
                    percdamp=.01, w_bit=w_bit, groupsize=q_config['q_group_size'], 
                    actorder='sx',
                    static_groups=False,
                    cfg=dim,
                )
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
            torch.cuda.empty_cache()
            gc.collect()
        block.load_state_dict(sd[best_dim])
        if adaptive:
            print(f">>> best_dim: (({best_dim}))")
        else:
            print(f">>> vanilla gptq, dim = ((oc)) ")
        gptq[n].free()
        del sd
        

    
@torch.inference_mode()
def run_gptq_ada(
    model, enc,
    w_bit, q_config,
    n_samples=512, seqlen=512,
    calib_data="pileval",
    adaptive=True,
):
    from ..utils.calib_data import get_calib_dataset
    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen)
    samples = torch.cat(samples, dim=0)
    print('sample shape', samples.shape)
    print('adaptive? ', adaptive)
    
    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    
    # get input and kwargs to layer 0
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
    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running GPTQ-ada..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # extract input and output features of all layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)
        def cache_output_hook(m, x, y, name, feat_dict):
            y = y[0]
            y = y.detach().cpu()
            feat_dict[name].append(y)

        input_feat = defaultdict(list)
        output_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name,
                                  feat_dict=input_feat)))
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_output_hook, name=name,
                                  feat_dict=output_feat)))
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        output_feat = {k: torch.cat(v, dim=0) for k, v in output_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()
        auto_dim_block(
            layer, layer_kwargs,
            w_bit=w_bit, q_config=q_config,
            input_feat=input_feat,
            output_feat=output_feat,
            adaptive=adaptive,
        )
        layer = layer.cpu()
        del input_feat
        
    return model
