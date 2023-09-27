#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export TRANSFORMERS_CACHE=/raid/hf_cache
export HF_DATASETS_CACHE=/raid/hf_cache

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate adadim

wbit=$2
gsize=$3

size=7B
size=13B
modelname=WizardMath-$size-V1.0
MODEL=WizardLM/$modelname
tasks=gsm8k

calib_data=pileval
calib_data=gsm8k

id=$modelname-w$wbit-g$gsize-gptq-calib=$calib_data
id=$modelname-w$wbit-g$gsize-gptq_ada-calib=$calib_data
echo "* $id"

# math
python -m adadim.entry --model_path $MODEL \
    --w_bit $wbit --q_group_size $gsize \
    --gptq_ada \
    --tasks $tasks --calib_data $calib_data \
    --output_path ./results/$id.json