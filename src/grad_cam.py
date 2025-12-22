import torch
import numpy as np
import cv2
from torchvision import models
import matplotlib.pyplot as plt

def generate_gradcam_heatmap(model, input_image, class_idx):
    model.eval()
    
    def hook_fn(module, input, output):
        global activations
        activations = output
    
    def grad_hook_fn(module, grad_in, grad_out):
        global gradients
        gradients = grad_out[0]

    activation_hook = model.resnet.layer4[1].register_forward_hook(hook_fn)
    gradient_hook = model.resnet.layer4[1].register_backward_hook(grad_hook_fn)
    
    output = model(input_image)
    loss = output[0, class_idx]
    loss.backward()

    activation_hook.remove()
    gradient_hook.remove()

    gradients = gradients.cpu().numpy()[0]
    activations = activations.cpu().numpy()[0]

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    cam /= np.max(cam)
    
    return cam
