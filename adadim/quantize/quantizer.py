import scipy
import torch
import torch.nn as nn
from tqdm import tqdm
import gc
    
# simulated quantization
def pseudo_quantize_tensor(w, n_bit=8,
                           zero_point=True, q_group_size=-1,
                           inplace=False,
                           get_scale_zp=False,
                           per_ic=False,
                           vector_quant=False, # w is a vector
                           ):
    # only one of them can be true; 
    assert per_ic & vector_quant == False 
    if per_ic:
        w = w.T #[OC, IC] -> [IC, OC]
    if vector_quant:
        w = w.unsqueeze(0)
    org_w_shape = w.shape

    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    else:
        w = w.reshape(-1, org_w_shape[-1])
    assert w.dim() == 2, w.shape
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    if vector_quant:
        w = w.squeeze(0)
    if per_ic:
        w = w.T #[IC, OC] -> [OC, IC]

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

@torch.inference_mode()
def pseudo_quantize_model_weight(
    model, w_bit, q_config, cfg='oc', w_update=False
):
    from .pre_quant import get_blocks, get_named_linears
    # import code; code.interact(local=dict(globals(), **locals()))
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            dim = 'ic' if 'ic' in cfg else 'oc'
            if i==0:
                print(f"quantizing {n} with cfg={dim} and wbit={w_bit} gsize={q_config['q_group_size']}")
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit,
                                                   per_ic = dim=='ic',
                                                   **q_config)
            m.cpu()
