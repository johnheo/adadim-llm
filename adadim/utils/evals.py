import os
import json
import torch
from lm_eval import evaluator
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils.modeling import get_balanced_memory
from adadim.utils.lm_eval_adaptor import LMEvalAdaptor
from adadim.utils.eval_mmlu import eval_mmlu
from adadim.utils.eval_gsm8k import gsm8k_test
from adadim.utils.eval_humaneval import humaneval_gen_ans

def eval(args, model, enc, max_memory):
    tot_results = {}
    task_names = args.tasks.split(",")

    if 'gsm8k' in task_names:
        args.tp = 1
        task_names.remove('gsm8k')
        print('* begin GSM8K evaluation...')
        acc = gsm8k_test(model, model_path=args.model_path,
                   batch_size=60, tensor_parallel_size=args.tp,
                   w_bit=args.w_bit, gsize=args.q_group_size,)
        tot_results['gsm8k'] = acc*100

    if 'humaneval' in task_names:
        args.tp = 1
        task_names.remove('humaneval')
        print('* begin HumanEval answer generation...')
        humaneval_gen_ans(model, model_path=args.model_path,
               output_path=args.response_path, start=0, end=164,
               temperature=0.0, N=1, max_len=2048, num_gpus=1,
               decoding_style='sampling', num_seqs_per_iter=1, overwrite=True,
               )
        print('* generated answers saved to ', args.response_path)

    # Move the model to GPU (as much as possible) for LM evaluation
    kwargs = {"max_memory": get_balanced_memory(model, max_memory if len(max_memory) > 0 else None)}
    device_map = infer_auto_device_map(
        model,
        # TODO: can we remove this?
        no_split_module_classes=[
            "OPTDecoderLayer", "LlamaDecoderLayer", "BloomBlock", "MPTBlock", "DecoderLayer"],
        **kwargs
    )

    model = dispatch_model(model, device_map=device_map)
    if 'csr' in task_names:
        csr = ['piqa', 'hellaswag', 'winogrande', 'arc_easy']
        task_names.remove('csr')
        task_names += csr

    if 'mmlu' in task_names:
        print('* begin MMLU evaluation...')
        mmlu_results = eval_mmlu(args, model, enc)
        tot_results["mmlu"] = mmlu_results
        task_names.remove('mmlu')

    if len(task_names) > 0:
        print('* begin Eval harness for ', task_names)

        lmeval_results = {}
        accs = []
        for task in task_names:
            if args.num_fewshot > -1:
                num_fewshot = args.num_fewshot
            else:
                num_fewshot = 0
            print(f'evaluating {task} with {num_fewshot} shot')
            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            result = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=[task],
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=num_fewshot,
            )
            print(evaluator.make_table(result))
            lmeval_results.update(result['results'])
            if task != 'wikitext':
                accs.append(result['results'][task]['acc'])
        
        if len(accs) > 0:
            lmeval_results['csr_avg'] = sum(accs) / len(accs)
        
        tot_results.update(lmeval_results)

        import pprint
        pprint.pprint(tot_results)

    if args.output_path is not None:
        print(args.output_path)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        tot_results["model"] = args.model_path
        tot_results["run_id"] = str(args.output_path.split('.')[0])
        
        with open(args.output_path, "w") as f:
            json.dump(tot_results, f, indent=2)