#!/bin/bash

# conda setup
conda env create -f adadim_env.yml
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate adadim

# mmlu data setup
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

# code data setup
git clone https://github.com/openai/human-eval
pip install -e human-eval