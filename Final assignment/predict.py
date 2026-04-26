"""
This script provides and example implementation of a prediction pipeline 
for a PyTorch U-Net model. It loads a pre-trained model, processes input 
images, and saves the predicted segmentation masks. 

You can use this file for submissions to the Challenge server. Customize 
the `preprocess` and `postprocess` functions to fit your model's input 
and output requirements.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, 
    ToImage, 
    Resize, 
    ToDtype, 
    Normalize,
    InterpolationMode,
)
import albumentations
from albumentations import (
    CLAHE, 
    Sharpen, 
    Compose as AlbumentationsCompose,
)

from model import Model

# Fixed paths inside participant container
# Do NOT chnage the paths, these are fixed locations where the server will 
# provide input data and expect output data.
# Only for local testing, you can change these paths to point to your local data and output folders.
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

#####################################################
### DEFINE POSTPROCESSING FUNCTIONS FOR IMAGE AUGMENTATIONS
#####################################################

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

def combined_augmentation(image, np_transforms, to_transforms):
        """
        Apply augmentations: first numpy-based (RandomFog, RandomRain),
        then torch-based (resize, normalize, etc.)
        """
        image_np = np.array(image, dtype=np.uint8)
        image_np = np_transforms(image=image_np)['image']
        image = torch.from_numpy(image_np).float() / 255.0  # Back to [0,1]
        image = image.permute(2, 0, 1)  # H,W,C → C,H,W
        
        image = to_transforms(image)
        
        return image

    # Create wrapper functions for the image transforms
    def base_image_transform(image):
        """Apply base preprocessing: CLAHE + Sharpen (numpy), then resize + normalize (torch)"""
        return combined_augmentation(image, base_img_numpy_transforms, base_img_torch_transform)

#####################################################
### DEFINE PREPROCESSING AND POSTPROCESSING
#####################################################

def preprocess(img: Image.Image) -> torch.Tensor:
    # Implement your preprocessing steps here
    # Apply base image transforms: numpy-based (CLAHE + Sharpen) then torch-based (resize + normalize)
    # Return a tensor suitable for model input
    img = base_image_transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Implement your postprocessing steps here
    # For example, resizing back to original shape, converting to color mask, etc.
    # Return a numpy array suitable for saving as an image
    pred_soft = nn.Softmax(dim=1)(pred)
    pred_max = torch.argmax(pred_soft, dim=1, keepdim=True)  # Get the class with the highest probability
    prediction = Resize(size=original_shape, interpolation=InterpolationMode.NEAREST)(pred_max)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch and channel dimensions if necessary

    return prediction_numpy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Model()
    state_dict = torch.load(
        MODEL_PATH, 
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        state_dict, 
        strict=True,  # Ensure the state dict matches the model architecture
    )
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))  # DO NOT CHANGE, IMAGES WILL BE PROVIDED IN THIS FORMAT
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # Forward pass
            pred = model(img_tensor)

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Create mirrored output folder
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Save predicted mask
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
