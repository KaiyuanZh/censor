"""
Defense methods.

Including:
- Additive noise
- Gradient clipping
- Gradient compression
- Representation perturbation
"""

# The code of defense is adapted from https://github.com/zhuohangli/GGL. 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import logging
import os
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(123)

# def additive_noise(input_gradient, std=0.1):
def additive_noise(model, input_gradient, save_dir, std=0.1):
    """
    Additive noise mechanism for differential privacy
    """
    gradient = [grad + torch.normal(torch.zeros_like(grad), std*torch.ones_like(grad)) for grad in input_gradient]
    return gradient


def gradient_clipping(input_gradient, bound=4):
    """
    Gradient clipping (clip by norm)
    """
    max_norm = float(bound)
    norm_type = 2.0 # np.inf
    device = input_gradient[0].device
    grad_tensor = [g.clone().cpu().detach() for g in input_gradient]
    
    if norm_type == np.inf:
        norms = [g.abs().max().to(device) for g in input_gradient]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type) for g in grad_tensor]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    
    gradient = [g.mul_(clip_coef_clamped.to(device)) for g in input_gradient]
    return gradient


def gradient_compression(input_gradient, percentage=10):
    """
    Prune by percentage
    """
    device = input_gradient[0].device
    gradient = [None]*len(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].clone().cpu().detach().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, percentage)    #取百分位数
        grad_tensor = torch.where(abs(input_gradient[i]) < thresh, 0, input_gradient[i])   #用阈值取梯度
        gradient[i] = torch.Tensor(grad_tensor).to(device)
    return gradient


def perturb_representation(input_gradient, model, ground_truth, pruning_rate=10):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device
    
    gt_data = ground_truth.clone()
    gt_data.requires_grad=True

    # register forward hook to get intermediate layer output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]
        return hook

    # for ResNet-18
    handle = model.fc.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)
    
    feature_graph = activation['flatten']

    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:,f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/((feature_graph.data[:,f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:,f] = 0
        
    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    
    print('Soteria mask: ', sum(mask))

    gradient = [grad for grad in input_gradient]
    # apply mask
    gradient[-2] = gradient[-2] * torch.Tensor(mask).to(device)
    
    handle.remove()
    
    return gradient

def generate_orthogonal_gradient(input_gradient):
    """
    Generate a new gradient that is orthogonal to the input_gradient.

    Args:
    - input_gradient: A list of gradients (tensors) for each layer of the model.

    Returns:
    - orthogonal_gradient: A list of gradients (tensors) that are orthogonal to the input_gradient.
    """
    orthogonal_gradient = []

    for grad in input_gradient:
        # Generate a random gradient with the same shape
        random_grad = torch.randn_like(grad)
        random_grad = random_grad / torch.norm(random_grad)

        # Flatten the gradients to treat them as vectors
        grad_flat = grad.flatten()
        random_grad_flat = random_grad.flatten()

        # Compute the projection of random_grad onto grad
        proj_scalar = torch.dot(random_grad_flat, grad_flat) / torch.norm(grad_flat)
        proj_vector = proj_scalar * grad_flat

        # Subtract the projection from random_grad to make it orthogonal
        orthogonal_grad_flat = random_grad_flat - proj_vector
        
        # Reshape back to the original shape and append to the list
        orthogonal_grad = orthogonal_grad_flat.view_as(grad)
        orthogonal_gradient.append(orthogonal_grad)

    return orthogonal_gradient

def normalize_orthogonal_gradient(ortho_gradient, original_gradient, fixed_norm=None):
    normalized_gradient = []
    for og, orig_g in zip(ortho_gradient, original_gradient):
        norm_og = torch.norm(og.flatten())
        norm_orig_g = fixed_norm if fixed_norm is not None else torch.norm(orig_g.flatten())
        normalized_g = (og / norm_og) * norm_orig_g if norm_og > 0 else og
        normalized_gradient.append(normalized_g)
    return normalized_gradient


def orthogonal_gradient(input_gradient, model, ground_truth, labels, trials, epsilon=1e-5, best_loss=float('inf')):
    criterion = nn.CrossEntropyLoss()
    best_gradient = None
    learning_rate = 1e-4

    original_params = [param.clone() for param in model.parameters()]  # Save original parameters outside the loop

    for trial in range(trials):
        model.train()
        noisy_gradient = generate_orthogonal_gradient(input_gradient)
        noisy_gradient = normalize_orthogonal_gradient(noisy_gradient, input_gradient)
        
        with torch.no_grad():
            for param, original, noise_grad in zip(model.parameters(), original_params, noisy_gradient):
                param.data.copy_(original - noise_grad * learning_rate)  # Update and evaluate in one step
            
            model.eval()
            new_loss = criterion(model(ground_truth), labels)

            # Logging less frequently
            if trial % 5 == 0:
                logger.info(f"{trial} trial loss: {new_loss}")

            # Revert to original parameters if needed
            if new_loss < best_loss:
                best_loss = new_loss
                best_gradient = [g.detach().clone() for g in noisy_gradient]  # Only clone when necessary
                logger.info(f'Gradient updated, Loss reduced to {best_loss}')

    # Restore original parameters
    for param, original in zip(model.parameters(), original_params):
        param.data.copy_(original)

    if best_gradient is not None:
        input_gradient = [g.clone() for g in best_gradient]

    return input_gradient, best_loss
