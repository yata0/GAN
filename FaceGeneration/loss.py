import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from torch import Tensor
from torch import autograd
loss_function = nn.BCEWithLogitsLoss()

def real_loss(predictions):
    # predictions = torch.squeeze(predictions)
    targets = torch.ones_like(predictions)
    if torch.cuda.is_available():
        targets = targets.cuda()
    loss = loss_function(predictions,targets)
    return loss

def fake_loss(predictions):
    targets = torch.zeros_like(predictions)
    if torch.cuda.is_available():
        targets = targets.cuda()
    loss = loss_function(predictions, targets)
    return loss

def WGAN_real_loss(predictions):
    return -torch.mean(predictions)

def WGAN_fake_loss(predictions):
    return torch.mean(predictions)

def WGAN_gp(real_samples, fake_samples, d_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = Tensor(np.random.random([real_samples.size(0),1,1,1])).to(device)
    interpolates = ((alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True))
    # if torch.cuda.is_available():
    #     interpolates = interpolates.cuda()
    d_interpolates = d_model(interpolates)
    fake = Variable(Tensor(real_samples.shape[0],1).fill_(1.0), requires_grad=False).to(device)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0),-1)
    gradient_penalty = ((gradients.norm(2,dim=1) - 1) ** 2).mean()
    return gradient_penalty


