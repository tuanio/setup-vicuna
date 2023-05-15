if [ ! -d "FastChat" ]; then
    git clone https://github.com/lm-sys/FastChat.git
fi

export WANDB_API_KEY=fedfe33b85e849d06384ed1cd7c3d356795b6060

wandb login fedfe33b85e849d06384ed1cd7c3d356795b6060

echo "Setup complete!"