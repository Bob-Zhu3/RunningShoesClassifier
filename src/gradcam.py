import numpy as np
import cv2
import torch

def generate_gradcam_heatmap(model, input_image, class_idx):
    model.eval()
    
    activations = None
    gradients = None

    def hook_fn(module, input, output):
        nonlocal activations
        activations = output
    
    def grad_hook_fn(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    activation_hook = model.resnet.layer4[1].register_forward_hook(hook_fn)
    gradient_hook = model.resnet.layer4[1].register_full_backward_hook(grad_hook_fn)
    
    output = model(input_image)
    model.zero_grad()
    loss = output[0, class_idx]
    loss.backward()

    activation_hook.remove()
    gradient_hook.remove()

    if gradients is None or activations is None:
        return None

    gradients = gradients.detach().cpu().numpy()[0]
    activations = activations.detach().cpu().numpy()[0]

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= np.min(cam)
    if np.max(cam) != 0:
        cam /= np.max(cam)
    
    return cam
