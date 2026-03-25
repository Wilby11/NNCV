"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
    ColorJitter,
    RandomApply,
    GaussianBlur,
    RandomAdjustSharpness
)

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    # Wilbur addition: Argument for loading a pre-trained model
    parser.add_argument("--pretrained-model", type=str, default=None, help="checkpoints/Baseline_model/best_model-epoch=0074-val_loss=0.29058183170855045.pt")
    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data; these transforms are applied to the images in the Cityscapes dataset
    img_transform = Compose([
        # Convert images to PIL format
        ToImage(),
        
        # Data augmentation transforms (before resizing!)
        ColorJitter(0.3, 0.3, 0.3, 0.05),  # varies brightness, contrast, saturation, and hue
        RandomApply([GaussianBlur(3)], p=0.2), # applies Gaussian blur with 20% probability
        RandomAdjustSharpness(2, p=0.2), # adjusts sharpness with 20% probability
        
        # Resize images to 320x224 using bilinear interpolation
        Resize((320, 224), interpolation=InterpolationMode.BILINEAR), # (width and height must be multiples of 16)
        # Convert images to torch float32 format and scale values to [0, 1]
        ToDtype(torch.float32, scale=True),
        # Normalize the images by subtracting the mean and dividing by the standard deviation
        # The mean and standard deviation are calculated from the entire Cityscapes dataset
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Target transform (mask)
    # These transforms are applied to the masks in the Cityscapes dataset
    # They are used to convert the masks to PIL format, resize them to 506x320,
    # and convert them to torch int64 format without scaling
    # The transformed masks are used in the training progress to compare to output images
    target_transform = Compose([
        ToImage(),  # Convert masks to PIL format
        Resize((320, 224), interpolation=InterpolationMode.NEAREST),  # Resize masks to 320x224 (width and height must be multiples of 16)
        ToDtype(torch.int64),  # Convert masks to torch int64 format without scaling
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
    args.data_dir,
    split="train",
    mode="fine",
    target_type="semantic",
    transform=img_transform,
    target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Wilbur addition: Load pre-trained model if specified
    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print(f"Loaded pre-trained model from {args.pretrained_model}")

    # Wilbur addition: Alternative loss function 'total_loss'
    def dice_loss(p, y, eps=1e-6):
        # p: (B,C,H,W), probs after softmax
        # y: (B,C,H,W), one-hot
        num = 2 * (p * y).sum(dim=(2,3))
        den = (p + y).sum(dim=(2,3)) + eps
        return 1 - (num / den).mean()
    def tv_loss(p):
        dx = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :]).mean()
        dy = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1]).mean()
        return dx + dy
        
    import torch.nn.functional as F
    def total_loss(logits, target, ignore_index=255, λ1=1.0, λ2=0.1):
        mask = (target != ignore_index)
        ce = F.cross_entropy(logits, target, ignore_index=ignore_index)

        target_onehot = F.one_hot(target.clamp(0,18), num_classes=logits.shape[1]).permute(0,3,1,2).float()
        pred = F.softmax(logits, dim=1)

        mask = mask.unsqueeze(1).float()
        pred_masked = pred * mask
        target_masked = target_onehot * mask

        dice = dice_loss(pred_masked, target_masked, eps=1e-6)
        tv = tv_loss(pred * mask)  # optional
        return ce + λ1 * dice + λ2 * tv

    criterion = total_loss

    # Define the loss function
    # criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)