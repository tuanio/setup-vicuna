# Prepare the Vicuna weights

13b

```
bash make_vicuna_weights.sh
```

# Training

## Support

- Just support these GPU systems: `H100`, `A100`, `RTX 3090`, `T4`, `RTX 2080`.

## Fine-tuning Vicuna-7B with Local GPUs

`torchrun` is a utility for running distributed PyTorch jobs.

### Options

- `--nproc_per_node`: Number of GPUs to use per node. Default is 1.
- `--master_port`: The port to use for communication between processes. Default is 20001.
- `FastChat/fastchat/train/train_mem.py`: The script to run using torchrun.
- `--model_name_or_path`: The name or path of the pretrained model to use.
- `--data_path`: The path to the training data.
- `--bf16`: Whether to use bfloat16 precision. Default is False.
- `--output_dir`: The directory to save output files.
- `--num_train_epochs`: The number of training epochs to run. Default is 3.
- `--per_device_train_batch_size`: The batch size per GPU for training. Default is 2.
- `--per_device_eval_batch_size`: The batch size per GPU for evaluation. Default is 2.
- `--gradient_accumulation_steps`: The number of batches to accumulate gradients over. Default is 16.
- `--evaluation_strategy`: The strategy for evaluating the model. Default is "no".
- `--save_strategy`: The strategy for saving the model. Default is "steps".
- `--save_steps`: The number of steps between each save. Default is 1200.
- `--save_total_limit`: The maximum number of checkpoints to keep. Default is 10.
- `--learning_rate`: The learning rate for training. Default is 2e-5.
- `--weight_decay`: The weight decay to use for training. Default is 0.0.
- `--warmup_ratio`: The ratio of steps to use for warming up the learning rate. Default is 0.03.
- `--lr_scheduler_type`: The type of learning rate scheduler to use. Default is "cosine".
- `--logging_steps`: The number of steps between each logging. Default is 1.
- `--fsdp`: The Fully Sharded Data Parallelism (FSDP) configuration. Default is "full_shard auto_wrap".
- `--fsdp_transformer_layer_cls_to_wrap`: The name of the Transformer layer class to wrap in FSDP. Default is "LlamaDecoderLayer".
- `--tf32`: Whether to use tf32 precision. Default is False.
- `--model_max_length`: The maximum length of input sequences for the model. Default is 2048.
- `--gradient_checkpointing`: Whether to use gradient checkpointing. Default is False.
- `--lazy_preprocess`: Whether to lazily preprocess the input data. Default is False.

Example command to train Vicuna-7B with 2 x A100 (40GB).
```
torchrun --nproc_per_node=2 --master_port=20001 FastChat/fastchat/train/train_lora.py \
    --model_name_or_path vicuna_weights/13b  \
    --data_path ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
    --bf16 True \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --torch_compile True
```

Only one machine
```
python FastChat/fastchat/train/train_lora.py \
    --model_name_or_path vicuna_weights/13b  \
    --data_path ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
    --bf16 True \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True
```

conda create -n dev -c pytorch-nightly -c nvidia -c pytorch -c conda-forge python=3.8 pytorch torchaudio cudatoolkit pandas numpy


# Serving

1. 
```
python3 -m fastchat.serve.controller
```

2. 
```
python3 -m fastchat.serve.model_worker --model-name 'demandgpt-v1.0' --model-path vicuna_weights/13b/
```

3.
```
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md

# UI

https://dev.to/deadwin19/deploy-nest-js-app-using-pm2-on-linux-ubuntu-server-88f#:~:text=Deploy%20Nest%20JS%20App%20using%20PM2%20on%20Linux,Project%29%20...%205%20Step%205%20%28Run%20Project%29%20


```
docker build -t chatgpt-ui .
docker run -e OPENAI_API_KEY=EMPTY -e DEFAULT_MODEL=demandgpt-v1.0 -e OPENAI_API_HOST=<API_HOST> -p 3000:3000 chatgpt-ui
```

## nginx running docker

```
sudo docker run --name ui_nginx -P -d nginx
```

# 
sudo apt install nginx

# content of 34.31.56.228
```
server {
    listen 80;
    server_name 34.31.56.228;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

sudo mv 34.31.56.228 /etc/nginx/sites-available/

sudo ln -s /etc/nginx/sites-available/34.31.56.228 /etc/nginx/sites-enabled/

## Test
sudo nginx -t
