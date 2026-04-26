# U-Net Semantic Segmentation on Cityscapes

This repository trains a U-Net model for semantic segmentation on Cityscapes. To reproduce the results, follow these steps.

## Setup

Clone repository from GitHub:

```bash
git clone https://<PAT>@github.com/<your-username>/NNCV.git
cd NNCV
```

Generate a Personal Access Token on GitHub if you don't have one (Settings → Developer Settings → Personal Access Tokens).

## Download Data and Container

Log into the SLURM cluster and run the download script once to get the Cityscapes dataset and Apptainer container:

```bash
chmod +x download_docker_and_data.sh
sbatch download_docker_and_data.sh
```

This creates:
- `data/cityscapes/` directory with training/validation/test splits
- `container.sif` file with all dependencies

## Configure Environment

Edit the `.env` file and add your Weights & Biases API key:

```env
WANDB_API_KEY=<your-key>
WANDB_DIR=/home/<username>/wandb
```

Get your API key from wandb.com.

## Train the Model

Submit a training job to SLURM:

```bash
chmod +x jobscript_slurm.sh
sbatch jobscript_slurm.sh
```

This trains the model with these settings:

- Batch size: 16
- Epochs: 200
- Learning rate: 0.0001 (halfed every 40 epochs)
- Seed: 42
- Experiment ID: "robust-U-Net-training"
- Model: U-Net with 3 input channels, 20 output classes, dropout=0.1
- Loss: Cross Entropy Loss + 0.5 * Dice Loss
- Optimizer: AdamW
- Augmentations: Augmentation pipeline applied with 50% probability during training, none during validation

The training script (`train.py`) validates on the Cityscapes validation set after each epoch. Checkpoints are saved to `checkpoints/robust-u-net-training/`.

Metrics are logged to Weights & Biases in real-time. Check your W&B dashboard at wandb.ai.

Monitor the job with:

```bash
tail -f slurm-<jobid>.out
```

## Get Your Best Model

After training the best model in the `checkpoints/robust-u-net-training/` folder. 

Copy it to the submission folder:

```bash
cp "checkpoints/robust-u-net-training/best_model-epoch=<N>.pt" "Final assignment/model.pt"
```

## Submit the Model

Build a Docker image locally with your best checkpoint:

```bash
docker build -t nncv-submission:latest -f "Final assignment/Dockerfile" "Final assignment"
```

Test it locally first:

```bash
mkdir local_data local_output
# Add test .png images to local_data/

docker run --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:latest
```

If it works do the following:

```bash
docker save -o nncv_submission.tar nncv-submission:latest
```

Upload `nncv_submission.tar` to the challenge servers (TU/e network only):
- Peak Performance: http://131.155.126.249:5001/
- Robustness: http://131.155.126.249:5002/
- Efficiency: http://131.155.126.249:5003/
- Out-of-Distribution: http://131.155.126.249:5004/

## Folder Structure

```
NNCV/
├── Final assignment/
│   ├── README.md
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── Dockerfile
│   ├── model.pt (after training)
│   └── checkpoints/
├── main.sh
├── jobscript_slurm.sh
├── download_docker_and_data.sh
├── .env
└── data/cityscapes/
```

## Notes

The model is a U-Net with dropout for regularization. Data augmentation is applied during training but not during validation. The void class (ID 255) is remapped to class 19 and trained as a regular class. Each image is resized to 256x512, normalized, and CLAHE and sharpening are applied as preprocessing steps. The training uses a combination of Cross Entropy Loss and Dice Loss to optimize both pixel-wise accuracy and overlap. The learning rate is halved every 40 epochs to help convergence.
