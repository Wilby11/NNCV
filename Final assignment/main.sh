wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 20 \
    --epochs 400 \
    --lr 0.0001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "moderate-class-weighting" \
    # --pretrained-model "checkpoints/Baseline_model/best_model-epoch=0074-val_loss=0.29058183170855045.pt"
    # --pretrained-model "checkpoints/new_attempt/best_model-epoch=0061-val_loss=1.1724237501621246.pt"
