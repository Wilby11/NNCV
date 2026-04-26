wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 400 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "final-training" \
    # --pretrained-model "..." # Optionally load some pretrained model
