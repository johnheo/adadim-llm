#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export TRANSFORMERS_CACHE=/raid/hf_cache
export HF_DATASETS_CACHE=/raid/hf_cache

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate adadim

# 7, 13, 30, 65, 70
wbit=$2
gsize=$3

size=7B
size=13B
modelname=WizardCoder-Python-$size-V1.0
MODEL=WizardLM/$modelname
tasks=humaneval


calib_data=pileval
calib_data=mbpp

id=$modelname-w$wbit-g$gsize-gptq-calib=$calib_data
id=$modelname-w$wbit-g$gsize-gptq_ada-calib=$calib_data
echo "* $id"

out_path=preds/$modelname/$id
mkdir -p ${out_path}
python -m adadim.entry --model_path $MODEL \
    --w_bit $wbit --q_group_size $gsize \
    --gptq_ada \
    --tasks $tasks --calib_data $calib_data \
    --output_path ./results/$id.json \
    --response_path $out_path

# eval outputs
echo 'Output path: '$out_path
python -m adadim.utils.process_humaneval \
    --path ${out_path} --out_path ${out_path}.jsonl --add_prompt
evaluate_functional_correctness ${out_path}.jsonl