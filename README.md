# Scaling with bottleneck layers

### Experiments to test the effect of scaling parameters for neural networks with a bottleneck layer.


For these experiments we have used data augmentation and determined the number of epochs for which the loss function stabilized --models are compared using this limit. We have also inspected training stability for varying hyperparameters of the learning rate scheduler with [cosine annealing](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).
 
 Current experiments evalutate:
 - The effect of initial conditions -- the variance scale of the weights pre-training on the training stability. Options include Xavier and He initializations. 
 - Number of parameters in the underparameterized regime (number of parameters << data set size).
 - Different bottleneck widths for fixed number of parameters.

References:
[Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)
[How to Start Training: The Effect of Initialization and Architecture](https://arxiv.org/abs/1803.01719)
[SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
