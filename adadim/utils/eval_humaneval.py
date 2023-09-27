# adopted from: https://github.com/nlpxucan/WizardLM/blob/main/WizardCoder/src/humaneval_gen_vllm.py
import os
from tqdm import tqdm
import torch
from human_eval.data import write_jsonl, read_problems

from adadim.quantize.rtn_ada import get_named_linears

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.parallel_utils.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def generate_prompt(input):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{input}

### Response:"""
    return INSTRUCTION

@torch.inference_mode()
def port_hf2vllm(hf_model, vllm_model):
    '''
        transfer Llama weights from hf format to vllm format. main changes are
        - [q, k, v] -> [qkv]
        - [up, gate] -> [up_gate]
    '''
    hf_sd = hf_model.state_dict()
    layers = vllm_model.llm_engine.workers[0].model.model.layers
    # iterate the hf model, get the right (module_name, weight) combo, then insert it to the vllm model 
    for i in tqdm(range(len(layers)), desc="transferring HF weights to VLLM"):
        named_linears = dict()
        named_linears.update(get_named_linears(layers[i], ColumnParallelLinear))
        named_linears.update(get_named_linears(layers[i], RowParallelLinear))
        named_linears.update(get_named_linears(layers[i], RMSNorm))
        for name, m in named_linears.items():
            if "qkv_" in name:
                accum = []
                for mod in ['q_', 'k_', 'v_']:
                    indexer = f"model.layers.{i}.self_attn.{mod}proj.weight"
                    accum.append(hf_sd[indexer])
                m.weight.data.copy_(torch.cat(accum, dim=0))
            elif "gate_up_proj" in name:
                accum = []
                for mod in ['gate_','up_']:
                    indexer = f"model.layers.{i}.mlp.{mod}proj.weight"
                    accum.append(hf_sd[indexer])
                m.weight.data.copy_(torch.cat(accum, dim=0))
            else:
                indexer = f"model.layers.{i}.{name}.weight"
                m.weight.data.copy_(hf_sd[indexer])

def humaneval_gen_ans(quant_model, model_path, lora="bigcode/starcoder", 
               output_path="./", start=0, end=164,
               temperature=0.8, N=200, max_len=512, num_gpus=2,
               decoding_style='sampling', num_seqs_per_iter=50, overwrite=False,
               ):

    problems = read_problems()

    task_ids = sorted(problems.keys())[start: end]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_len)
    print('sampling =====', sampling_params)

    print('* Max Mem before VLLM',torch.cuda.max_memory_reserved()/1024/1024/1024, 'GB')
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus) # in gpu
    print('* Max Mem after VLLM',torch.cuda.max_memory_reserved()/1024/1024/1024, 'GB')

    port_hf2vllm(quant_model, llm)
    print('* Max Mem after Porting',torch.cuda.max_memory_reserved()/1024/1024/1024, 'GB')
    print('* sanity check')
    print(quant_model.model.layers[0].self_attn.q_proj.weight[0, :10])
    print(llm.llm_engine.workers[0].model.model.layers[0].self_attn.qkv_proj.weight[0,:10])

    print(f"Loaded {model_path}.")
    print(f'create output dir {output_path}')
    os.makedirs(output_path, exist_ok=True)
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = output_path + '/{}.jsonl'.format(start + i)
        if os.path.exists(output_file) and not overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        if decoding_style == 'sampling':
            loops = int(N // num_seqs_per_iter)
        else:
            loops = 1
        
        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]
            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]
                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)