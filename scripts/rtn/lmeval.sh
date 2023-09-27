#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
export TRANSFORMERS_CACHE=/raid/hf_cache
export HF_DATASETS_CACHE=/raid/hf_cache

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate adadim

ver=2
size=7b
# size=13b
# size=70b

# ver=1
# size=30b

modelname=llama-$size
if [ $ver = "1" ]
then
    MODEL=huggyllama/llama-$size
    modelname=llama-$size
else
    MODEL=meta-llama/Llama-2-$size-hf
    modelname=llama-2-$size
fi


wbit=3
gsize=128
tasks=mmlu,csr
id=$modelname-w$wbit-g$gsize-rtn_ada

echo "* $id"
python -m adadim.entry --model_path $MODEL \
    --w_bit $wbit --q_group_size $gsize \
    --rtn_ada \
    --tasks $tasks \
    --output_path ./results/$id.json
