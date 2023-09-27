
import os
import gc
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from adadim.utils.parallel import auto_parallel
from adadim.utils.evals import eval

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of the hf model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=-1)
# model config
parser.add_argument('--parallel', action='store_true',
                    help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument('--max_memory', type=str, nargs='*',
                    help="List of device_id:max_memory pairs to be parsed into a dictionary; " \
                        + "Example: 0:10GiB 1:10GiB cpu:30GiB; " \
                        + "mode details here: " \
                        + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling")
parser.add_argument('--auto_parallel', action='store_true',
                    help="automatically set parallel and batch_size")
# quantization config
parser.add_argument('--w_bit', type=int, default=16)
parser.add_argument('--q_group_size', type=int, default=128)
parser.add_argument('--no_zero_point', action='store_true',
                    help="disable zero_point")
# mine
parser.add_argument('--calib_data', type=str, default='pileval', 
                    choices=['pileval', 'gsm8k', 'humaneval', 'mbpp'])
parser.add_argument("--response_path", default=None, type=str,
                    help="path to the code generation file")
parser.add_argument('--rtn_ada', action='store_true',
                    help="run adaptive RTN")
parser.add_argument('--gptq_ada', action='store_true',
                    help="run adaptive GPTQ")
parser.add_argument("--icoc_cfg", default="awq", type=str)
args = parser.parse_args()

max_memory = [v.split(':') for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k):v for k,v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization

}
print("Quantization config:", q_config)

# build model and tokenizer
def build_model_and_enc(model_path, cache_dir=None):
    print(f"* Building model {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, use_auth_token=True)
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    # Init model on CPU:
    kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True, "use_safetensors": False, "use_auth_token": True}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True, **kwargs).eval()
    return model, enc

def main():
    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)

    if args.w_bit < 16:
        if args.gptq_ada:
            from adadim.update.gptq_ada import run_gptq_ada
            print('model maxpos: ', model.config.max_position_embeddings)
            q_model = run_gptq_ada(
                model, enc,
                w_bit=args.w_bit, q_config=q_config,
                n_samples=256,
                seqlen=512,
                calib_data=args.calib_data,
                adaptive='ada' in args.output_path,
            )
        elif args.rtn_ada:
            from adadim.quantize.rtn_ada import run_rtn_ada
            q_model = run_rtn_ada(
                model, enc,
                w_bit=args.w_bit, q_config=q_config,
                n_samples=256,
                seqlen=512,
                calib_data=args.calib_data,
                adaptive='ada' in args.output_path,
            )
    else: # fp16
        q_model = model
    
    # eval
    gc.collect()
    torch.cuda.empty_cache()
    if args.tasks is not None:
        print('\n>>> evaluting: ', args.output_path)
        eval(args, q_model, enc, max_memory)

if __name__ == '__main__':
    main()
