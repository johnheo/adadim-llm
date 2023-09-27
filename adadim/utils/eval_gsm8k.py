# adopted from: https://github.com/nlpxucan/WizardLM/blob/main/WizardMath/inference/gsm8k_inference.py
import re
import sys
import tqdm
import torch
import jsonlines

from fraction import Fraction

from adadim.quantize.rtn_ada import get_named_linears

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.parallel_utils.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

INVALID_ANS = "[invalid]"
MAX_INT = sys.maxsize
VERBOSE = False

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

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
    for i in tqdm.tqdm(range(len(layers)), desc="transferring HF weights to VLLM"):
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
                # import code; code.interact(local=dict(globals(), **locals()))
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

def gsm8k_test(quant_model, model_path, data_path='data/gsm8k_test.jsonl', start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1,
               w_bit=3, gsize=128):
    
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1024, stop=stop_tokens)
    print('sampling =====', sampling_params)

    print('* Max Mem before VLLM',torch.cuda.max_memory_reserved()/1024/1024/1024, 'GB')
    llm = LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=tensor_parallel_size) # in gpu
    print('* Max Mem after VLLM',torch.cuda.max_memory_reserved()/1024/1024/1024, 'GB')
    
    port_hf2vllm(quant_model, llm)

    print('* sanity check')
    print(quant_model.model.layers[0].self_attn.q_proj.weight[0, :10])
    print(llm.llm_engine.workers[0].model.model.layers[0].self_attn.qkv_proj.weight[0,:10])

    result = []
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)
    
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('ACC ====', acc, 'len invalid outputs ====', len(invalid_outputs))
    print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)
    return acc