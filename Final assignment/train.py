"""
This script is used to train the U-net model as specified in `model.py`. U-net is trained as follows:
1. Each epoch a batch of training images is used.
2. Each image in the batch is preprocessed, and a augmentation pipeline is applied with a 50% probability.
3. The model is trained using cross-entropy loss combined with dice loss, and class weights are used to handle class imbalance.
4. Validation is performed without augmentation.
"""
import os
from argparse import ArgumentParser
import math
import signal
import random
import numpy as np
import warnings
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (Compose, Normalize, Resize, ToImage, ToDtype, InterpolationMode,
    ColorJitter, RandomApply, GaussianBlur, RandomAdjustSharpness)
from model import Model
import torch.nn.functional as F
import albumentations
from albumentations import (
    RandomShadow, 
    HueSaturationValue, 
    RandomBrightnessContrast, 
    GaussNoise, 
    CLAHE, 
    Sharpen, 
    RandomFog, 
    RandomRain,  
    Compose as AlbumentationsCompose,
)

#########################################################################################
### I MOVED ARGUMENT PARSING TO THE TOP OF THE FILE BECAUSE I NEEDED THE SEED ARGUMENT 
### IN THE SHADOW AUGMENTATION CLASS, WHICH IS DEFINED BELOW
#########################################################################################

def get_args_parser():
    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    # Wilbur addition: Argument for loading a pre-trained model if you specify it in main.sh
    parser.add_argument("--pretrained-model", type=str, default=None, help="Load the weights of a previously trained model")
    return parser

# Note: Seed initialization happens inside main after parsing args

#########################################################################################
### MAPPING CLASS IDs TO TRAIN IDs
#########################################################################################

# Cityscapes uses ID 255 for void class, but we remap all void labels to 
# class 19 instead so the model predicts it as a real class instead of ignoring it. Note: the labels are 0,...,19.
void_train_id = 19
id_to_trainid = { cls.id: (void_train_id if cls.train_id == 255 else cls.train_id)
                  for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])
# Mapping train IDs to color
train_id_to_color = { (void_train_id if cls.train_id == 255 else cls.train_id): cls.color
                       for cls in Cityscapes.classes}
# Assign black to void class
train_id_to_color[void_train_id] = (0, 0, 0)  

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image

#########################################################################################
### AUGMENTATION SETUP: PROBABILISTIC AUGMENTATION APPLIED DURING TRAINING
#########################################################################################

def main(args):
    # Initialize wandb for logging
    wandb.init( project="5lsm0-cityscapes-segmentation",  # Project name in wandb
                name=args.experiment_id,  # Experiment name in wandb
                config=vars(args),)  # Save hyperparameters
    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducibility. If you add other sources of randomness (NumPy, random), make sure to set their seeds as well
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)    # Sets the seed for NumPy RNG (for ShadowAugmentation's randomness)
    torch.manual_seed(seed) # Sets the seed for torch RNG (for ColorJitter, RandomApply, etc.)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #########################################################################################
    ### DEFINE THE IMAGE TRANSFORMS: BOTH PREPROCESSING AND AUGMENTATIONS
    #########################################################################################

    # Define the transforms to apply to the data; these transforms are applied to the images in the Cityscapes dataset
    base_img_numpy_transforms = AlbumentationsCompose([
        CLAHE(clip_limit=(1,4), tile_grid_size=(8,8), p=1), # By far the most effective augmentation (visually at least), so this should be applied to every single image
        Sharpen(alpha=(0.7,0.8), p=1), # Sharpen the image to enhance edges and details, which can help the model learn better features from the augmented images
    ], is_check_shapes=False)
    
    base_img_torch_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    aug_img_numpy_transforms = AlbumentationsCompose([
        RandomFog(alpha_coef=0.16, fog_coef_range=(0.3, 0.4), p=0.05),
        RandomRain(slant_range=(-20,20), blur_value=10, brightness_coefficient=0.75, drop_length=15, rain_type='default', p=0.10),
        RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(2,2), shadow_dimension=4, p=0.25),
        HueSaturationValue(hue_shift_limit=50, sat_shift_limit=30, val_shift_limit=(-15,50), p=0.25),  # Adjust hue, saturation, and value (values is brightness and shouldn't be too low to avoid making the image too dark for the shadow augmentation to work effectively)
        RandomBrightnessContrast(brightness_limit=(-0.1,0.5), contrast_limit=(-0.1,0.5), p=0.2),  # Adjust brightness and contrast with 30% limit (again do not want to let the image get too dark)
        GaussNoise(std_range=(0.1, 0.2), p=0.1),  # Add Gaussian noise (I keep std between 0.1 and 0.2 since the images are already low resolution)
    ], is_check_shapes=False)

    aug_img_torch_transform = Compose([
        ToImage(),
        Resize((256, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Target transform (mask):
    # These transforms are applied to the masks in the Cityscapes dataset
    # They are used to convert the masks to PIL format, resize them to 256x512 using nearest neighbor interpolation (to preserve class labels),
    # and convert them to torch int64 format without scaling. The transformed masks are used in the training progress to compare to output images
    target_transform = Compose([
        ToImage(),  # Convert masks to PIL format
        Resize((256, 512), interpolation=InterpolationMode.NEAREST),  # Resize masks to 256x512 (width and height must be multiples of 16)
        ToDtype(torch.int64),  # Convert masks to torch int64 format without scaling
    ])

    def combined_augmentation(image, np_transforms, to_transforms):
        """
        Apply augmentations: first numpy-based (RandomFog, RandomRain),
        then torch-based (resize, normalize, etc.)
        """
        # Convert PIL to numpy array (H,W,C format for albumentations)
        image_np = np.array(image, dtype=np.uint8)

        # Apply numpy-based augmentations (expects H,W,C)
        image_np = np_transforms(image=image_np)['image']
        
        # Convert back to tensor (C,H,W format for torch)
        image = torch.from_numpy(image_np).float() / 255.0  # Back to [0,1]
        image = image.permute(2, 0, 1)  # H,W,C → C,H,W
        
        # Apply torch-based transforms
        image = to_transforms(image)
        
        return image

    # Create wrapper functions for the image transforms
    def base_image_transform(image):
        """Apply base preprocessing: CLAHE + Sharpen (numpy), then resize + normalize (torch)"""
        return combined_augmentation(image, base_img_numpy_transforms, base_img_torch_transform)
    
    def aug_image_transform(image):
        """Apply augmented transforms: weather effects + color (numpy), then resize + normalize (torch)"""
        return combined_augmentation(image, aug_img_numpy_transforms, aug_img_torch_transform)
    
    def probabilistic_aug_image_transform(image, augmentation_probability=0.5):
        """Apply base transforms, then apply aug_transforms with a given probability"""
        # First apply base transforms (CLAHE + Sharpen + torch transforms)
        image_np = np.array(image, dtype=np.uint8)
        image_np = base_img_numpy_transforms(image=image_np)['image']
        image = torch.from_numpy(image_np).float() / 255.0
        image = image.permute(2, 0, 1)
        
        # With probability, apply augmentations (all of them or none)
        if random.random() < augmentation_probability:
            # Convert back to PIL for numpy augmentations
            image = image.permute(1, 2, 0) * 255.0
            image = Image.fromarray(image.numpy().astype(np.uint8))
            image_np = np.array(image, dtype=np.uint8)
            image_np = aug_img_numpy_transforms(image=image_np)['image']
            image = torch.from_numpy(image_np).float() / 255.0
            image = image.permute(2, 0, 1)
        
        # Apply torch-based transforms (resize + normalize)
        image = base_img_torch_transform(image)
        return image

    #########################################################################################
    ### DEFINE THE DATA LOADERS
    #########################################################################################

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=probabilistic_aug_image_transform,  # Apply probabilistic augmentation to training images
        target_transform=target_transform,
    )

    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=base_image_transform,  # Apply only base transforms to validation images (no augmentation)
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

    #########################################################################################
    ### Main: COMPUTE CLASS WEIGHTS FOR HANDLING CLASS IMBALANCE
    #########################################################################################

    def compute_class_weights(dataloader, num_classes, device, max_images=500):
        """
        Compute class weights using moderate weighting for underrepresented classes.
        Uses sqrt of inverse frequency instead of raw inverse frequency to avoid extreme weights.
        """
        class_counts = torch.zeros(num_classes, device=device)
        total_pixels = 0
        samples_processed = 0
        
        print(f"Computing class weights (sampling first {max_images} images)...")
        for batch_idx, (_, labels) in enumerate(dataloader):
            labels = convert_to_train_id(labels)
            labels = labels.squeeze(1).long().to(device)
            
            samples_in_batch = labels.shape[0]
            samples_processed += samples_in_batch
            
            for class_id in range(num_classes):
                class_counts[class_id] += (labels == class_id).sum().item()
            total_pixels += labels.numel()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {samples_processed} images...")
            
            if samples_processed >= max_images:
                print(f"  Reached {samples_processed} images, stopping.")
                break
        
        # Moderate weighting: use sqrt of inverse frequency
        # This provides smoother weight scaling compared to raw inverse frequency
        # weight = sqrt(total / (num_classes * class_count))
        # Avoid division by zero for empty classes
        class_weights = torch.ones(num_classes, device=device)
        for class_id in range(num_classes):
            if class_counts[class_id] > 0:
                # Sqrt of inverse frequency provides moderate weighting
                class_weights[class_id] = torch.sqrt(torch.tensor(
                    total_pixels / (num_classes * class_counts[class_id]), 
                    device=device
                ))
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum() * num_classes
        
        print("Class weights computed (using moderate sqrt-based weighting):")
        for class_id in range(num_classes):
            print(f"  Class {class_id}: {class_weights[class_id]:.4f} (count: {class_counts[class_id].item()})")
        
        return class_weights
    
    class_weights = compute_class_weights(train_dataloader, 20, device, max_images=500)

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=20,  # 19 Cityscapes classes plus void
        dropout_p=0.1,  # Dropout probability
    ).to(device)

    #########################################################################################
    ### Main: INITIALIZE WEIGHTS FROM A PRE-TRAINED MODEL IF SPECIFIED, OTHERWISE USE U-NET INITIALIZATION
    #########################################################################################

    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        # Freeze early layers
        for param in model.inc.parameters():
            param.requires_grad = False
        for param in model.down1.parameters():
            param.requires_grad = False
        print("First layers will remain frozen. .")
    else:
        def init_weights(m):
            """ Initialize weights according to the U-Net paper:
            Draw from Gaussian with std = sqrt(2/N).
            """
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                std = math.sqrt(2.0 / n)
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        model.apply(init_weights)

    #########################################################################################
    ### Main: DEFINE THE LOSS FUNCTION
    ######################################################################################### 
    
    def dice_loss(p, y, eps=1e-6):
        # p: (B,C,H,W), probs after softmax
        # y: (B,C,H,W), one-hot
        num = 2 * (p * y).sum(dim=(2,3))
        den = (p + y).sum(dim=(2,3)) + eps
        return 1 - (num / den).mean()
        
    
    def segmentation_loss(logits, target, class_weights=None):
        # Combine cross-entropy (with class weights for imbalance) and dice loss
        ce = F.cross_entropy(logits, target, weight=class_weights)
        target_onehot = F.one_hot(target, num_classes=logits.shape[1]).permute(0,3,1,2).float()
        pred = F.softmax(logits, dim=1)
        dice = dice_loss(pred, target_onehot, eps=1e-6)
        # focal = FocalLoss(alpha=1, gamma=2)(logits, target)

        return ce + 0.5 * dice

    #########################################################################################
    ### Main: START TRAINING LOOP
    #########################################################################################

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)  # this scheduler halfes the learning rate every 40 epochs
    best_valid_loss = float('inf')
    current_best_model_path = None
    
    # Setup signal handler to convert SIGTERM to an exception
    def handle_sigterm(signum, frame):
        """Convert SIGTERM signal to SystemExit exception for proper cleanup"""
        print(f"\nReceived SIGTERM signal (HPC job termination)...")
        raise SystemExit(f"SIGTERM received (signal {signum})")
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    
    try:
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

                loss = segmentation_loss(outputs, labels, class_weights=class_weights)
                
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
                    loss = segmentation_loss(outputs, labels, class_weights=class_weights)
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
            scheduler.step()
            
        print("Training complete!")

        # Save the model
        torch.save(
            model.state_dict(),
            os.path.join(
                output_dir,
                f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
            )
        )
    
    except (KeyboardInterrupt, SystemExit):
        print(f"\n{'='*60}")
        print("Training interrupted! Saving current model weights...")
        print(f"{'='*60}")
        interrupt_model_path = os.path.join(
            output_dir,
            f"interrupted_model-epoch={epoch:04}.pt"
        )
        torch.save(model.state_dict(), interrupt_model_path)
        print(f"Model weights saved to: {interrupt_model_path}")
        raise
    
    finally:
        print("Cleaning up and closing W&B session...")
        wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)