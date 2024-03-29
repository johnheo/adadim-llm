# adopted from: https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_mmlu.py
import os
import pandas as pd
import numpy as np
import argparse
import datasets
import torch

from typing import List
from tqdm import tqdm
from transformers.trainer_utils import set_seed


'''
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
'''
DATA_PATH = "./data/mmlu/data"

TASK_NAME_MAPPING = {'stem': ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning'],
 'Humanities': ['formal_logic', 'high_school_european_history', 'high_school_us_history', 'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law', 'world_religions'],
 'other': ['business_ethics', 'college_medicine', 'human_aging', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine', 'virology', 'global_facts', 'clinical_knowledge'],
 'social': ['econometrics', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology', 'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy']}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]


def load_models_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.generation import GenerationConfig
    config = AutoConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, config=config,
                                                  device_map="balanced", trust_remote_code=True).eval()
    if 'qwen' in args.checkpoint_path.lower():
        model.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    
    return model, tokenizer


def format_example(line, include_answer=True):
    example = 'Question: ' + line['question']
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'
            
    if include_answer:
        example += '\nAnswer: ' + line["answer"] + '\n\n'
    else:
        example += '\nAnswer:'
    return example


def generate_few_shot_prompt(k, subject, dev_df):

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()
    
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))

    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt


def get_logits(args, tokenizer, model, inputs: List[str]):
    input_ids = tokenizer(inputs, padding=False)['input_ids']
    input_ids = torch.tensor(input_ids, device=model.device)

    if input_ids.shape[1] > args.max_seq_len:
        input_ids = input_ids[:, input_ids.shape[1]-args.max_seq_len+1:]
    tokens = {'input_ids': input_ids}

    outputs = model(input_ids)['logits']
    logits = outputs[:, -1, :]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs, {'tokens': tokens}


@torch.inference_mode()
def eval_subject(args,
        model,
        tokenizer,
        subject_name,
        test_df,
        k=5,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        **kwargs
):
    result = []
    score = []

    few_shot_prompt = generate_few_shot_prompt(
        k, subject_name, dev_df) if few_shot else []
    all_probs = {'prob_A': [], 'prob_B': [], 'prob_C': [], 'prob_D': []}
    if args.debug: print(f"few_shot_prompt: {few_shot_prompt}")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), disable=True):
        question = format_example(row, include_answer=False)
        full_prompt = few_shot_prompt + question
        
        output, input_info = get_logits(args, tokenizer, model, [full_prompt])
        assert output.shape[0] == 1
        logits = output.flatten()
        softval = torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer(" A")['input_ids']][-1],
                        logits[tokenizer(" B")['input_ids']][-1],
                        logits[tokenizer(" C")['input_ids']][-1],
                        logits[tokenizer(" D")['input_ids']][-1],
                    ]
                ),
                dim=0,
            )
        if softval.dtype in {torch.bfloat16, torch.float16}:
            softval = softval.to(dtype=torch.float32)
        probs = softval.detach().cpu().numpy()

        for i, choice in enumerate(choices):
            all_probs[f'prob_{choice}'].append(probs[i])
        # import code; code.interact(local=locals())
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        if 'answer' in row:
            correct = 1 if pred == row['answer'] else 0
            score.append(correct)
            if args.debug: print(f'{question} pred: {pred} ans: {row["answer"]}')
        result.append(pred)

    if save_result_dir:
        test_df['model_output'] = result
        for i, choice in enumerate(choices):
            test_df[f'prob_{choice}'] = (all_probs[f'prob_{choice}'])
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(os.path.join(
            save_result_dir, f'{subject_name}_result.csv'), encoding="utf-8", index=False)

    return score


def cal_mmlu(res):
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.
    cnt = 0
    hard_cnt = 0
    hard_acc_sum = 0.

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.
        acc_norm_sum_dict[class_] = 0.
        cnt_dict[class_] = 0.

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    print('\n\n\n', 'total cnt:', cnt, '\n')
    results = {}
    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print('%s ACC: %.2f ' % (
                k, acc_sum_dict[k] / cnt_dict[k] * 100))
            results[k] = acc_sum_dict[k] / cnt_dict[k] * 100
    print('=> AVERAGE ACC:%.2f ' % (acc_sum / cnt * 100))
    results['average'] = acc_sum / cnt * 100
    return results

    

def eval_mmlu(args, model, tokenizer, 
              eval_data_path=DATA_PATH,
              debug=False):
    # model, tokenizer = load_models_tokenizer(args)
    args.max_seq_len = 2048 if 'v2' not in args.model_path else 4096
    args.debug = debug

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        # val_file_path = os.path.join(eval_data_path, 'val', f'{subject_name}_val.csv')
        dev_file_path = os.path.join(eval_data_path, 'dev', f'{subject_name}_dev.csv')
        test_file_path = os.path.join(eval_data_path, 'test', f'{subject_name}_test.csv')
        # val_df = pd.read_csv(val_file_path, names=['question','A','B','C','D','answer'])
        dev_df = pd.read_csv(dev_file_path, names=['question','A','B','C','D','answer'])
        test_df = pd.read_csv(test_file_path, names=['question','A','B','C','D','answer'])

        score = eval_subject(args, model, tokenizer, subject_name, test_df, dev_df=dev_df, k=5, few_shot=True,
                            #  save_result_dir=f"outs/mmlu_eval_result",
                            )
        dev_result[subject_name] = score
        acc = sum(score)/len(score)
        # print("* Average accuracy {:.3f} - {}".format(acc, subject_name))
    results = cal_mmlu(dev_result)

    return results