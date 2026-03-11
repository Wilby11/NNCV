## Things I need to do when opening this project
Run the command `git pull` to pull all updates made to the course's repository.

## Github
I forked the course's repository. I then made a clone on my laptop, and I made a clone on Snellius. I will be working in the cloned repository on Snellius when I am training the neural network, and I will be working on the local clone when I am playing around with the model design. When I `git push`, I push to my forked repository. I need to add certain files in the `.gitignore`, such as the `container.sif` file which contains my data, because we don't want to have that on our laptop, and also not on our fork. 

## WandB
This website tracks the training of my neural network. It has graphs that show validation and training losses per iteration, and even shows the 64 segementation images that each batch gets (currently 64, but can change this later). Follow a training session here:
```
https://wandb.ai/w-p-v-lierop-eindhoven-university-of-technology/5lsm0-cityscapes-segmentation/workspace?nw=nwuserwpvlierop
```

## SLURM
The Slurm Workload Manager, formerly known as Simple Linux Utility for Resource Management (SLURM), or simply Slurm, is a free and open-source job scheduler for Linux and Unix-like kernels, used by many of the world's supercomputers and computer clusters. Its primary job includes allocating exclusive and/or non-exclusive access to computer nodes to users for some duration of time so they can perform work.

## Primary objective
- Submit a baseline model before Tuesday

## What I should do / investigate
- Investigate the dataset: find out how many images, which labels, how should we label them?
- Investigate the current model: what learning parameters are we using? How is the loss defined? How big is the network?

## Ideas
- Use the loss function from Computer Vision assignment to better take various lightings into account and increase robustness
