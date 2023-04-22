"""This file contains a set of decorators to help with the training of PyTorch models.

To use these decorators in your PyTorch code, import the decorators and add them before the relevant functions.
eg, 
@device_decorator
@tqdm_decorator
def train_one_epoch(model, dataloader, criterion, optimizer):
    # Training code here

Decorator headings are listed in the CONTENTS section below 
Description of each decorator and usage is on the comment above its definition
Dependencies are listed under ### IMPORTS 
"""

""" CONTENTS

# GENERAL TRAINING UTILITIES 
@seed_decorator(seed=42)
@device_decorator

# PROFILING
@tqdm_decorator
@tensorboard_decorator(log_dir="./logs")
@pytorch_profiler_decorator(use_cuda=True)
@activation_stats_decorator()

# GENERAL LAYER MANIPULATION
@model_freezing_decorator(layers_to_freeze=["conv1"], layers_to_unfreeze=["conv2"])
@noise_injection_decorator(scale=0.01)

# GRADIENTS
@gradient_analysis_decorator
@gradient_clipping_decorator(clip_value=5.0)
@gradient_accumulation_decorator(accumulation_steps=4)

# BATCH NORM
@bn_folding_decorator
@freeze_bn_layers_decorator

# DROPOUT
@dynamic_dropout_decorator(initial_p=0.5, rate=0.1)

"""


### IMPORTS
import torch
import torch.nn as nn
import torch.profiler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from tqdm import trange


### DEFINITIONS START HERE 
"""Set a random seed for reproducibility 

@seed_decorator(seed=42)
def train(self, loader, optimizer, criterion):
    # Define the training loop here
"""
def seed_decorator(seed):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            return func(*args, **kwargs)
        return wrapper
    return decorator


"""Ensure that the model and input tensors are on the same device (CPU/GPU)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define model layers here

    @device_decorator
    def forward(self, x):
        # Define the forward pass here
"""
def device_decorator(func):
    def wrapper(*args, **kwargs):
        model = args[0]
        device = next(model.parameters()).device
        new_args = []
        new_kwargs = {}

        for arg in args[1:]:
            if isinstance(arg, torch.Tensor):
                new_args.append(arg.to(device))
            else:
                new_args.append(arg)

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                new_kwargs[key] = value.to(device)
            else:
                new_kwargs[key] = value

        return func(model, *new_args, **new_kwargs)
    return wrapper


"""Display a progress bar for training loop 

@tqdm_decorator(epochs=10)
def train(model, loader, optimizer, criterion):
    # Define the training loop here
"""
def tqdm_decorator(epochs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            model = args[0]
            for epoch in trange(epochs, desc="Training", unit="epoch"):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


"""Log metrics to TensorBoard

@tensorboard_decorator(log_dir="./logs")
def train(model, loader, optimizer, criterion, step=0):
    # Define the training loop here
"""
def tensorboard_decorator(log_dir):
    writer = SummaryWriter(log_dir)
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            step = kwargs.get("step", 0)

            if isinstance(result, dict):
                for key, value in result.items():
                    writer.add_scalar(f"{key}", value, step)

            return result
        return wrapper
    return decorator


"""PyTorch profiler for memory/time spent in each operation
Identify performance bottlenecks

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define model layers here

    @pytorch_profiler_decorator(use_cuda=False)
    def forward(self, x):
        # Define the training loop here

"""
def pytorch_profiler_decorator(use_cuda=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with torch.autograd.profiler.profile(use_cuda=use_cuda, profile_memory=True, record_shapes=True) as prof:
                result = func(*args, **kwargs)
            print(prof)
            return result
        return wrapper
    return decorator


"""Print activation statistics (mean and std) of each layer during training
Implemented for ReLu only, modify as needed

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Define model layers here

    @activation_stats_decorator()
    def forward(self, x):
        # Define the forward pass here
"""
def activation_stats_decorator():
    activation_stats = {}

    def decorator(func):
        def wrapper(*args, **kwargs):
            model = args[0]
            register_hooks(model)
            return func(*args, **kwargs)

        def register_hooks(model):
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.ReLU):
                    module.register_forward_hook(forward_hook(name))

        def forward_hook(name):
            def hook(module, input, output):
                mean = output.data.mean().item()
                std = output.data.std().item()
                if name not in activation_stats:
                    activation_stats[name] = {'mean': [], 'std': []}
                activation_stats[name]['mean'].append(mean)
                activation_stats[name]['std'].append(std)
                print(f"Activation stats for layer {name}: Mean = {mean:.4f}, Std = {std:.4f}")
            return hook

        return wrapper
    return decorator


"""Freeze/unfreeze specific layers in model during training

@model_freezing_decorator(layers_to_freeze=["conv1"], layers_to_unfreeze=["conv2"])
def train(model, loader, criterion, optimizer):
    # Training code here
"""
def model_freezing_decorator(layers_to_freeze=None, layers_to_unfreeze=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            model = args[0]

            if layers_to_freeze:
                for name, param in model.named_parameters():
                    if any(layer_name in name for layer_name in layers_to_freeze):
                        print(name, "frozen...")
                        param.requires_grad = False

            if layers_to_unfreeze:
                for name, param in model.named_parameters():
                    if any(layer_name in name for layer_name in layers_to_unfreeze):
                        print(name, "unfrozen...")
                        param.requires_grad = True

            return func(*args, **kwargs)
        return wrapper
    return decorator


"""Inject noise into model weights during training 

@noise_injection_decorator(scale=0.01)
def train(model, dataloader, criterion, optimizer):
"""
def noise_injection_decorator(scale=0.01):
    def decorator(func):
        def wrapper(*args, **kwargs):
            model = args[0]

            for name, param in model.named_parameters():
                if 'weight' in name:
                    print("... adding noise to", name)
                    noise = torch.randn_like(param) * scale
                    param.data.add_(noise)

            return func(*args, **kwargs)

        return wrapper
    return decorator


"""Print the gradient statistics (mean, min, max) for each layer

@gradient_analysis_decorator
def train(model, loader, criterion, optimizer):
    # Training code here
"""
def gradient_analysis_decorator(func):
    def wrapper(*args, **kwargs):
        model = args[0]
        handles = register_gradient_hooks(model)
        result = func(*args, **kwargs)
        remove_gradient_hooks(handles)
        return result

    def register_gradient_hooks(model):
        handles = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handle = module.register_backward_hook(hook)
                handles.append(handle)

        return handles

    def hook(module, input, output):
        if module.weight.grad is not None:
            grad_data = module.weight.grad.data
            print(f"{module} - Grad Mean: {np.mean(grad_data.numpy()):.4f}, "
                    f"Min: {np.min(grad_data.numpy()):.4f}, "
                    f"Max: {np.max(grad_data.numpy()):.4f}")
            
    def remove_gradient_hooks(handles):
        for handle in handles:
            handle.remove()

    return wrapper


"""Clip the gradients during training to prevent exploding gradients

@gradient_clipping_decorator(clip_value=1.0)
def train(model, loader, criterion, optimizer):
    # Training code here
"""
def gradient_clipping_decorator(clip_value):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            model = args[0]
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            return result
        return wrapper
    return decorator


"""Accumulate gradients over multiple mini-batches before optimizer step
Helpful for large batch sizes that give memory errors

@gradient_accumulation_decorator(accumulation_steps=4)
def train_one_batch(model, dataloader, optimizer, criterion, step=0):
"""
def gradient_accumulation_decorator(accumulation_steps):
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = args[2]
            loss = func(*args, **kwargs)

            if kwargs.get("step", 0) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            return loss
        return wrapper
    return decorator


"""Batchnorm folding 
Note: needs modules initialized in sequential order 
@bn_folding_decorator
def evaluate(model, loader, criterion):
    # Evaluation code here
"""
def bn_folding_decorator(func):
    def wrapper(*args, **kwargs):
        model = args[0]
        model = fold_bn(model)
        return func(model, *args[1:], **kwargs)

    def fold_bn(model):
        folded_model = nn.Sequential()

        def merge_conv_bn(conv, bn):
            merged_conv = nn.utils.fuse_conv_bn_eval(conv, bn)
            return merged_conv
        
        prev_module = None
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d) and isinstance(prev_module, nn.Conv2d):
                folded_conv = merge_conv_bn(prev_module, module)
                folded_model.add_module(prev_name, folded_conv)
            else:
                if prev_module is not None:
                    folded_model.add_module(prev_name, prev_module)
                prev_module = module
                prev_name = name

        if prev_module is not None:
            folded_model.add_module(prev_name, prev_module)

        return folded_model

    return wrapper


"""Freeze the parameters of all batchnorm layers

@freeze_bn_layers_decorator
def fine_tune(model, loader, criterion, optimizer):
"""
def freeze_bn_layers_decorator(func):
    def wrapper(*args, **kwargs):
        model = args[0]

        for module in model.modules():     
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(False)
        return func(*args, **kwargs)
    return wrapper


"""Dynamically adjust dropout probability during training

@dynamic_dropout_decorator(initial_p=0.5, rate=0.1)
def train(model, loader, criterion, optimizer, epoch=10):
    # Training code here
"""
def dynamic_dropout_decorator(initial_p=0.5, rate=0.1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            model = args[0]
            epoch = kwargs.get('epoch', 0)
            next_p = initial_p * (1 - rate) ** epoch

            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = next_p

            print(f"Dropout probability adapted to {next_p}")
            return func(*args, **kwargs)

        return wrapper
    return decorator