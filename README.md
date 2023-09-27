# Rethinking Channel Dimensions to Isolate Outliers for Low-bit Weight Quantization of Large Language Models

## Set up

Environment & Data:
`sh setup.sh`

For math QA data, download the raw file from [here](https://github.com/nlpxucan/WizardLM/blob/main/WizardMath/data/gsm8k_test.jsonl) and place it in the `./data/` folder

Set your path to HF cache directory in each of the bash files below. E.g.,

- `export TRANSFORMERS_CACHE=/username/my_hf_cache`
- `export HF_DATASETS_CACHE=/username/my_hf_cache`

## Usage

Running RTN-ada

- To evaluate MMLU and Common Sense Reasoning (CSR):
`sh scripts/rtn/lmeval.sh`
- To evaluate math reasoning:
`sh scripts/rtn/math.sh`
- To evaluate code generation:
`sh scripts/rtn/code.sh`

Running GPTQ-ada

- To evaluate MMLU and Common Sense Reasoning (CSR):
`sh scripts/gptq/lmeval.sh`
- To evaluate math reasoning:
`sh scripts/gptq/math.sh`
- To evaluate code generation:
`sh scripts/gptq/code.sh`

## Acknowledgements

This code base is expanded upon wonderful works from

- https://github.com/mit-han-lab/llm-awq
- https://github.com/IST-DASLab/gptq
- https://github.com/nlpxucan/WizardLM
- https://github.com/QwenLM/Qwen