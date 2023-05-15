#!/bin/bash

while getopts ":m:" opt; do
  case $opt in
    m)
      model="$OPTARG"
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ $model == "7b" ]]; then
    python3 -m fastchat.model.apply_delta --base-model-path decapoda-research/llama-7b-hf \
        --target-model-path vicuna_weights/7b --delta-path lmsys/vicuna-7b-delta-v1.1 

    echo "Model output weights to this path: vicuna_weights/13b"
elif [[ $model == "13b" ]]; then
    python -m fastchat.model.apply_delta --base-model-path decapoda-research/llama-13b-hf \
        --target-model-path vicuna_weights/13b --delta-path lmsys/vicuna-13b-delta-v1.1

    echo "Model output weights to this path: vicuna_weights/13b"
else
    echo "Invalid model option. Please choose between 7b and 13b."
    exit 1
fi
